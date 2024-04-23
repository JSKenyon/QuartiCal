from collections import defaultdict

from distributed import SchedulerPlugin
from dask.core import reverse_dict
from dask.base import tokenize
from dask.order import ndependencies


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(ScatterSolvers(**kwargs), idempotent=True)


def unravel_deps(hlg_deps, name, unravelled_deps=None):
    """Recursively construct a set of all dependencies for a specific task."""

    if unravelled_deps is None:
        unravelled_deps = set()

    for dep in hlg_deps[name]:
        unravelled_deps |= {dep}
        unravel_deps(hlg_deps, dep, unravelled_deps)

    return unravelled_deps


def get_node_depths(dependencies, root_nodes, metrics):

    node_depths = {}

    for k in dependencies.keys():
        # Get dependencies per node.
        deps = unravel_deps(dependencies, k)
        # Associate nodes with root nodes.
        roots = root_nodes & deps
        offset = metrics[k][-1]
        node_depths[k] = \
            max(metrics[r][-1] - offset for r in roots) if roots else 0

    return node_depths


class AutoRestrictor(SchedulerPlugin):

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None,
                     **kw):
        """Processes dependencies to assign tasks to specific workers."""

        workers = list(scheduler.workers.keys())
        n_worker = len(workers)

        tasks = scheduler.tasks
        dependencies = kw["dependencies"]
        dependents = reverse_dict(dependencies)

        _, total_dependencies = ndependencies(dependencies, dependents)
        # TODO: Avoid calling graph metrics.
        metrics = graph_metrics(dependencies, dependents, total_dependencies)

        # Terminal nodes have no dependents, root nodes have no dependencies.
        # Horizontal partition nodes are initialized as the terminal nodes.
        part_nodes = {k for (k, v) in dependents.items() if not v}
        root_nodes = {k for (k, v) in dependencies.items() if not v}

        # Figure out the depth of every task. Depth is defined as maximum
        # distance from a root node. TODO: Optimize get_node_depths.
        node_depths = get_node_depths(dependencies, root_nodes, metrics)
        max_depth = max(node_depths.values())

        # If we have fewer partition nodes than workers, we cannot utilise all
        # the workers and are likely dealing with a reduction. We work our way
        # back through the graph, starting at the deepest terminal nodes, and
        # try to find a depth at which there was enough work to utilise all
        # workers.
        while (len(part_nodes) < n_worker) & (max_depth > 0):
            _part_nodes = part_nodes.copy()
            for pn in _part_nodes:
                if node_depths[pn] == max_depth:
                    part_nodes ^= set((pn,))
                    part_nodes |= dependencies[pn]
            max_depth -= 1
            if max_depth <= 0:
                return  # In this case, there in nothing we can do - fall back.

        part_roots = {}
        part_dependencies = {}
        part_dependents = {}

        for pn in part_nodes:
            # Get dependencies per partition node.
            part_dependencies[pn] = unravel_deps(dependencies, pn)
            # Get dependents per partition node.
            part_dependents[pn] = unravel_deps(dependents, pn)
            # Associate partition nodes with root nodes.
            part_roots[pn] = root_nodes & part_dependencies[pn]

        # Create a unique token for each set of partition roots. TODO: This is
        # very strict. What about nodes with very similar roots? Tokenization
        # may be overkill too.
        root_tokens = {
            tokenize(
                *sorted(v, key=lambda x: str(x))
            ): v for v in part_roots.values()
        }
        hash_map = defaultdict(set)
        group_offset = 0

        # Associate partition roots with a specific group if they are not a
        # subset of another, larger root set.
        for k, v in root_tokens.items():
            if any(v < vv for vv in root_tokens.values()):  # Strict subset.
                continue
            else:
                hash_map[k] |= set([group_offset])
                group_offset += 1

        # If roots were a subset, they should share the group of their
        # superset/s.
        for k, v in root_tokens.items():
            if not v:  # Special case - no dependencies. Handled below.
                continue
            shared_roots = \
                {kk: None for kk, vv in root_tokens.items() if v < vv}
            if shared_roots:
                hash_map[k] = \
                    set().union(*[hash_map[kk] for kk in shared_roots.keys()])

        task_groups = defaultdict(set)

        for pn in part_nodes:

            pdp = part_dependencies[pn]
            pdn = part_dependents[pn]

            if pdp:
                groups = hash_map[
                    tokenize(*sorted(part_roots[pn],  key=lambda x: str(x)))
                ]
            else:  # Special case - no dependencies.
                groups = {group_offset}
                group_offset += 1

            for g in groups:
                task_groups[g] |= pdp | pdn | {pn}

        worker_loads = {wkr: 0 for wkr in workers}

        for task_group in task_groups.values():

            assignee = min(worker_loads, key=worker_loads.get)
            worker_loads[assignee] += len(task_group)

            for task_name in task_group:
                try:
                    task = tasks[task_name]
                except KeyError:  # Keys may not have an assosciated task.
                    continue
                if task.worker_restrictions is None:
                    task.worker_restrictions = set()
                task.worker_restrictions |= {assignee}
                task.loose_restrictions = False


# NOTE: The graph metrics function has been copied directly from
# dask/order.py. This function was removed upstream but is currently
# critical for the scheduler plugin. This is a temporary work around
# while we consider revised strategies. Original version appears at:
# https://github.com/dask/dask/blob/9f7d55705f0eb74a172f5fdc3690aa65fbe21762/dask/order.py
def graph_metrics(dependencies, dependents, total_dependencies):
    """Useful measures of a graph used by ``dask.order.order``"""

    result = {}
    num_needed = {k: len(v) for k, v in dependents.items() if v}
    current = []
    current_pop = current.pop
    current_append = current.append
    for key, deps in dependents.items():
        if not deps:
            val = total_dependencies[key]
            result[key] = (1, val, val, 0, 0)
            for child in dependencies[key]:
                num_needed[child] -= 1
                if not num_needed[child]:
                    current_append(child)

    while current:
        key = current_pop()
        parents = dependents[key]
        if len(parents) == 1:
            (parent,) = parents
            (
                total_dependents,
                min_dependencies,
                max_dependencies,
                min_heights,
                max_heights,
            ) = result[parent]
            result[key] = (
                1 + total_dependents,
                min_dependencies,
                max_dependencies,
                1 + min_heights,
                1 + max_heights,
            )
        else:
            (
                total_dependents_,
                min_dependencies_,
                max_dependencies_,
                min_heights_,
                max_heights_,
            ) = zip(*(result[parent] for parent in dependents[key]))
            result[key] = (
                1 + sum(total_dependents_),
                min(min_dependencies_),
                max(max_dependencies_),
                1 + min(min_heights_),
                1 + max(max_heights_),
            )
        for child in dependencies[key]:
            num_needed[child] -= 1
            if not num_needed[child]:
                current_append(child)
    return result


class ScatterSolvers(SchedulerPlugin):

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None,
                     **kw):
        """Processes dependencies to assign tasks to specific workers."""

        workers = list(scheduler.workers.keys())
        n_worker = len(workers)

        tasks = scheduler.tasks
        dependencies = kw["dependencies"]
        dependents = reverse_dict(dependencies)

        solver_tasks = [
            k for k in dependencies.keys() if k[0].startswith('solver_wrapper')
        ]

        solver_task_groups = []

        for st in solver_tasks:

            solver_dependencies = unravel_deps(dependencies, st)
            solver_dependendents = unravel_deps(dependents, st)

            solver_task_groups.append(
                {st} | solver_dependencies | solver_dependendents
            )

        worker_loads = {wkr: 0 for wkr in workers}

        for task_group in solver_task_groups:

            assignee = min(worker_loads, key=worker_loads.get)
            worker_loads[assignee] += len(task_group)

            for task_name in task_group:
                try:
                    task = tasks[task_name]
                except KeyError:  # Keys may not have an assosciated task.
                    continue
                if task.worker_restrictions is None:
                    task.worker_restrictions = set()
                task.worker_restrictions |= {assignee}
                task.loose_restrictions = False