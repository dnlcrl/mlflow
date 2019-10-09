import click
import mlflow
from mlflow.utils import cli_args


@click.group("redisai")
def commands():
    """
    Serve models on RedisAI.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("deploy")
@click.option('--project-root-directory', help='project directory, where model\'s files defitions resies', required=True)
@click.option("--model-key", help="Model key", required=True)
@click.option("--input-shape", help="space separated integers in string production input shape for exporting model jit trace. e.g --input-shape '1 1 4'", required=True, type=str)
@cli_args.MODEL_URI
@click.option("--host", "-h", default='localhost', help="RedisAI Server host address")
@click.option("--port", "-p", default=6379, help="RedisAI Server port", type=int)
@click.option("--db", default=None, help="RedisAI db")
@click.option("--password", default=None, help="RedisAI password")
@click.option("--socket-timeout", default=None, help="RedisAI socket_timeout", type=int)
@click.option("--socket-connect-timeout", default=None, help="RedisAI Server socket_connect_timeout", type=int)
@click.option("--socket-keepalive", default=None, help="RedisAI Server posocket_keepalivert")
@click.option("--socket-keepalive-options", default=None, help="RedisAI Server socket_keepalive_options")
@click.option("--connection-pool", default=None, help="RedisAI Server connection_pool")
@click.option("--unix-socket-path", default=None, help="RedisAI Server unix_socket_path")
@click.option("--encoding", default=u'utf-8', help="RedisAI Server encoding")
@click.option("--encoding-errors", default=u'strict', help="RedisAI Server encoding_errors")
@click.option("--charset", default=None, help="RedisAI Server charset")
@click.option("--errors", default=None, help="RedisAI Server errors", type=bool)
@click.option("--decode-responses", default=False, help="RedisAI Server decode_responses", type=bool)
@click.option("--retry-on-timeout", default=False, help="RedisAI Server retry_on_timeout", type=bool)
@click.option("--ssl", default=False, help="RedisAI Server ssl")
@click.option("--ssl-keyfile", default=None, help="RedisAI Server ssl_keyfile")
@click.option("--ssl-certfile", default=None, help="RedisAI Server ssl_certfile")
@click.option("--ssl-cert-reqs", default=u'required', help="RedisAI Server ssl_cert_reqs")
@click.option("--ssl-ca-certs", default=None, help="RedisAI Server ssl_ca_certs")
@click.option("--max-connections", default=None, help="RedisAI Server max_connections", type=int)
@click.option("--backend", default='CPU', help="RedisAI backend (CPU or GPU)", type=str)


def deploy(project_root_directory, model_key, input_shape, model_uri, host, port, db, password, socket_timeout, socket_connect_timeout, 
    socket_keepalive, socket_keepalive_options, connection_pool, unix_socket_path, encoding, 
    encoding_errors, charset, errors, decode_responses, retry_on_timeout, ssl, ssl_keyfile, 
    ssl_certfile, ssl_cert_reqs, ssl_ca_certs, max_connections, backend):
    """
    Deploy model on RedisAI.
    """
    mlflow.redisai.deploy(project_root_directory=project_root_directory, model_key=model_key, input_shape=input_shape, model_uri=model_uri,
        host=host, port=port, db=db, password=password, socket_timeout=socket_timeout, socket_connect_timeout=socket_connect_timeout, 
        socket_keepalive=socket_keepalive, socket_keepalive_options=socket_keepalive_options, connection_pool=connection_pool, unix_socket_path=unix_socket_path, encoding=encoding, 
        encoding_errors=encoding_errors, charset=charset, errors=errors, decode_responses=decode_responses, retry_on_timeout=retry_on_timeout, ssl=ssl, ssl_keyfile=ssl_keyfile, 
        ssl_certfile=ssl_certfile, ssl_cert_reqs=ssl_cert_reqs, ssl_ca_certs=ssl_ca_certs, max_connections=max_connections, backend=backend)

