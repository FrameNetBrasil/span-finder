# span-finder FN-Br

This is a fork of the original [span-finder](https://github.com/hiaoxui/span-finder) with adaptations made by FrameNet Brasil.

The model is dockerized to make it easier to use.

## Building the container

Make sure you have docker installed and then run at the root of the repo:

```shell
docker build . -t "<tag-name>"
```

A good ``<tag-name>`` to make things easier is **lome**.

## Training

When the docker image is built, it already configures relevant paths for the training procedure. To run the default training, two volumes must be mapped:

- **Data**: this is where the training data is located, the folder must contain the files ``train.jsonl``, ``dev.jsonl``, ``test.jsonl`` and ``ontology``;
- **Checkpoint**: this is the folder in the host machine where the model checkpoints will be saved (_There's no need to create this folder, only map it, the docker process will create it automatically_).

Suppose these two folders are ``data`` and ``model-checkpoint`` in the current folder. The run command should be:

```shell
sudo docker run -v $(pwd)/data/:/srv/data -v $(pwd)/model-checkpoint:/srv/checkpoint/ --gpus all -it "lome"
```

Once inside the container, you can run the following training command:

```shell
allennlp train -s $CHECKPOINT_PATH --include-package sftp config/fn.jsonnet
```

It is important to note that ``$CHECKPOINT_PATH`` is already set on the image. To check all the default configurations for paths, check [.env.default](.env.default). Just be careful: if any of those paths are changed, the mapping of volumes need to change as well. The most important value is ``CUDA``. By default, the process will try to use **cuda:0**. To train on CPU, use the following command when running the container:

```shell
sudo docker run -e CUDA=[-1] -v $(pwd)/data/:/srv/data -v $(pwd)/model-checkpoint:/srv/checkpoint/ -it "lome"
```

or set the variable inside the container:

```shell
export CUDA=[-1]
```

Finally, to make changes to training parameters, make a copy of [config/fn.jsonnet](config/fn.jsonnet), change parameters and map the new file to the container using ``-v`` when running. Then only the last part of the training command needs to be changed:

```shell
allennlp train -s $CHECKPOINT_PATH --include-package sftp <path-to-jsonnet>
```

After the training is done, the ckeckpoints should be available at ``model-checkpoint``. If docker was executed as **sudo**, just change permissions to see the results.

```shell
sudo chown -R $USER ./model-checkpoint/
```