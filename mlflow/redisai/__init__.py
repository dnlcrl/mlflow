"""
The ``mlflow.sagemaker`` module provides an API for deploying MLflow models to Amazon SageMaker.
"""
from __future__ import print_function

import os
from subprocess import Popen, PIPE, STDOUT
from six.moves import urllib
import sys
import tarfile
import logging
import time

import mlflow
import mlflow.version
from mlflow import pyfunc, mleap
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE, IO_ERROR
from mlflow.utils import get_unique_resource_id
from mlflow.utils.logging_utils import eprint
from mlflow.models.container import SUPPORTED_FLAVORS as SUPPORTED_DEPLOYMENT_FLAVORS
from mlflow.models.container import DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import redis
import torch
import mlflow.pytorch
from mlflow.utils.model_utils import _get_flavor_configuration
import mlflow.pyfunc.utils as pyfunc_utils
import io

# DEFAULT_IMAGE_NAME = "mlflow-pyfunc"
DEPLOYMENT_MODE_ADD = "add"
DEPLOYMENT_MODE_REPLACE = "replace"
FLAVOR_NAME = "pytorch"
DEPLOYMENT_MODES = [
    DEPLOYMENT_MODE_ADD,
    DEPLOYMENT_MODE_REPLACE
]

# IMAGE_NAME_ENV_VAR = "MLFLOW_SAGEMAKER_DEPLOY_IMG_URL"
# Deprecated as of MLflow 1.0.
# DEPRECATED_IMAGE_NAME_ENV_VAR = "SAGEMAKER_DEPLOY_IMG_URL"

# DEFAULT_BUCKET_NAME_PREFIX = "mlflow-sagemaker"

# DEFAULT_SAGEMAKER_INSTANCE_TYPE = "ml.m4.xlarge"
# DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1

_logger = logging.getLogger(__name__)

# _full_template = "{account}.dkr.ecr.{region}.amazonaws.com/{image}:{version}"

def set_redismodel(redis_client, graph_blob, graphkey, backend):
    return redis_client.execute_command('AI.MODELSET', graphkey, 'TORCH', backend, graph_blob)

def _get_preferred_deployment_flavor(model_config):
    """
    Obtains the flavor that MLflow would prefer to use when deploying the model.
    If the model does not contain any supported flavors for deployment, an exception
    will be thrown.

    :param model_config: An MLflow model object
    :return: The name of the preferred deployment flavor for the specified model
    """
    if mleap.FLAVOR_NAME in model_config.flavors:
        return mleap.FLAVOR_NAME
    elif pyfunc.FLAVOR_NAME in model_config.flavors:
        return pyfunc.FLAVOR_NAME
    else:
        raise MlflowException(
            message=(
                "The specified model does not contain any of the supported flavors for"
                " deployment. The model contains the following flavors: {model_flavors}."
                " Supported flavors: {supported_flavors}".format(
                    model_flavors=model_config.flavors.keys(),
                    supported_flavors=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=RESOURCE_DOES_NOT_EXIST)


def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is a supported deployment flavor
    and is contained in the specified model. If one of these conditions
    is not met, an exception is thrown.

    :param model_config: An MLflow Model object
    :param flavor: The deployment flavor to validate
    """
    if flavor not in SUPPORTED_DEPLOYMENT_FLAVORS:
        raise MlflowException(
            message=(
                "The specified flavor: `{flavor_name}` is not supported for deployment."
                " Please use one of the supported flavors: {supported_flavor_names}".format(
                    flavor_name=flavor,
                    supported_flavor_names=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=INVALID_PARAMETER_VALUE)
    elif flavor not in model_config.flavors:
        raise MlflowException(
            message=("The specified model does not contain the specified deployment flavor:"
                     " `{flavor_name}`. Please use one of the following deployment flavors"
                     " that the model contains: {model_flavors}".format(
                        flavor_name=flavor, model_flavors=model_config.flavors.keys())),
            error_code=RESOURCE_DOES_NOT_EXIST)


def deploy(project_root_directory, model_key, input_shape, model_uri, host, port, db, password, socket_timeout, socket_connect_timeout, 
    socket_keepalive, socket_keepalive_options, connection_pool, unix_socket_path, encoding, 
    encoding_errors, charset, errors, decode_responses, retry_on_timeout, ssl, ssl_keyfile, 
    ssl_certfile, ssl_cert_reqs, ssl_ca_certs, max_connections, backend):

    redis_client =  redis.StrictRedis(**dict(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=socket_keepalive,
            socket_keepalive_options=socket_keepalive_options,
            connection_pool=connection_pool,
            unix_socket_path=unix_socket_path,
            encoding=encoding,
            encoding_errors=encoding_errors,
            charset=charset,
            errors=errors,
            decode_responses=decode_responses,
            retry_on_timeout=retry_on_timeout,
            ssl=ssl,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            max_connections=max_connections,
        ))
    device = 'cpu' if backend == 'CPU' else 'cuda' 
    model_path = _download_artifact_from_uri(model_uri)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    code_subpath = pyfunc_conf.get(pyfunc.CODE)
    if code_subpath is not None:
        pyfunc_utils._add_code_to_system_path(
            code_path=os.path.join(model_path, code_subpath))
    pytorch_conf = _get_flavor_configuration(model_path=model_path, flavor_name=FLAVOR_NAME)
    if torch.__version__ != pytorch_conf["pytorch_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed PyTorch version '%s'",
            pytorch_conf["pytorch_version"], torch.__version__)

    sys.path.append(project_root_directory)
    model =  mlflow.pytorch._load_model(path=model_uri + '/data').to(device)

    x = torch.ones(list(map(int, input_shape.split(' ')))).to(device)
    traced_net = torch.jit.trace(model, x)
    blob_stream = io.BytesIO()
    torch.jit.save(traced_net, blob_stream)
    blob_stream.seek(0)
    set_redismodel(redis_client, blob_stream.read(), model_key, backend)


def _load_pyfunc_conf(model_path):
    """
    Loads the `python_function` flavor configuration for the specified model or throws an exception
    if the model does not contain the `python_function` flavor.
    :param model_path: The absolute path to the model.
    :return: The model's `python_function` flavor configuration.
    """
    model_path = os.path.abspath(model_path)
    model = Model.load(os.path.join(model_path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise MlflowException(
                message=("The specified model does not contain the `python_function` flavor. This "
                         " flavor is required for model deployment required for model deployment."),
                error_code=INVALID_PARAMETER_VALUE)
    return model.flavors[pyfunc.FLAVOR_NAME]

