import subprocess
import multiprocessing
import time

def run_command(cmd):
    import tensorflow as tf
    import tensorflow_hub as hub
    import gc

    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    hub_handle = "./film_1"
    model = hub.load(hub_handle)
    print("Hello, world!")
    print(result.stdout.decode('utf-8'))
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == '__main__':
    #run_command('ls -l')
    p = multiprocessing.Process(target=run_command, args=('ls -l',))
    p.start()
    p.join()
    