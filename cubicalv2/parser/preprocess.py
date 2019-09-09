# -*- coding: utf-8 -*-
from loguru import logger
import re
import dask.array as da


def preprocess_opts(opts):
    """Preprocesses the namespace/dictionary given by opts.

    Given a namespace/dictionary of options, this should verify that that
    the options can be understood. Some options specified as strings need
    further processing which may include the raising of certain flags.

    Args:
        opts: A namepsace/dictionary of options.

    Returns:
        Namespace: An updated namespace object.
    """

    opts._model_columns = []
    opts._sky_models = {}
    opts._predict = False
    opts._internal_recipe = {}

    model_recipes = opts.input_model_recipe.split(":")

    # TODO: Consider how to implement operations on model sources. This will
    # require a stacking and rechunking approach in addition to adding and
    # subtracting visibilities.

    # TODO: Repeated .lsm files overwrite dictionary contents. This needs to
    # be fixed.

    for recipe_index, model_recipe in enumerate(model_recipes):

        opts._internal_recipe[recipe_index] = []

        # A raw string is required to avoid insane escape characters. Splits
        # on understood operators, ~ for subtract, + for add.

        ingredients = re.split(r'([\+~])', model_recipe)

        # Behaviour of re.split guaratees every second term is either a column
        # or .lsm.

        for ingredient in ingredients:

            if ingredient == "":
                continue
            elif ingredient in "~+":
                operation = da.add if ingredient == "+" else da.subtract
                opts._internal_recipe[recipe_index].append(operation)

            elif ".lsm.html" in ingredient:
                filename, _, tags = ingredient.partition("@")
                tags = tags.split(",")
                opts._sky_models[filename] = tags
                opts._predict = True
                # opts._internal_recipe[recipe_index].append({filename: tags})
                opts._internal_recipe[recipe_index].append(filename)

            elif ingredient != "":
                opts._model_columns.append(ingredient)
                opts._internal_recipe[recipe_index].append(ingredient)

    logger.info("The following model sources were obtained from "
                "--input-model-recipe: \n"
                "   Columns: {} \n"
                "   Sky Models: {}",
                opts._model_columns,
                list(opts._sky_models.keys()))

    if opts._predict:
        logger.info("Enabling prediction step.")
