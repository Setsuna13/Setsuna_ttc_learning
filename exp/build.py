#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import importlib
import os
import sys
sys.path.insert(0, os.getcwd())


def get_exp_by_file(exp_file):
    try:
        exp_path = os.path.abspath(os.path.normpath(exp_file))
        module_parts = [os.path.splitext(os.path.basename(exp_path))[0]]
        package_dir = os.path.dirname(exp_path)
        import_root = package_dir
        while os.path.isfile(os.path.join(package_dir, "__init__.py")):
            module_parts.insert(0, os.path.basename(package_dir))
            import_root = os.path.dirname(package_dir)
            package_dir = import_root
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
        current_exp = importlib.import_module(".".join(module_parts))
        exp = current_exp.Exp()
    except Exception as exc:
        raise ImportError(
            "{} doesn't contain a loadable class named 'Exp': {}".format(
                exp_file, exc
            )
        )
    return exp


def get_exp(exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
