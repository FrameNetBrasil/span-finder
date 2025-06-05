# span-finder FN-Br

This is a fork of the original [span-finder](https://github.com/hiaoxui/span-finder) with adaptations made by FrameNet Brasil.

The model is dockerized to make it easier to use.

## Building the container

Make sure you have docker installed and then run at the root of the repo:

```shell
docker build . -t "<tag-name>"
```

A good ``<tag-name>`` to make things easier is **span-finder**.

To prevent errors when downloading the spaCy model, use:

```shell
docker build --network=host . -t "<tag-name>"
```

## Training

### (optional) Building the dataset

To train a new model with the most up-to-date data, the [build-fnbr-data](./tools/build-fnbr-dataset/) tool can be used. It is used to merge annotation from multiple sources, deanonymize them if required and convert annotation from character to token spans.

The tool expects a list of sources as a comma-separated string that may contain values ``fn17``, ``fnbr`` and ``ehr``, representing Berkeley FrameNet's 1.7 release, "normal" FrameNet Brasil annotation data and annotation of EHRs (eletronic health records) that are anonymized. If ``fn17`` is in sources, the ``--fn17_path`` parameter must be specified pointing to a folder with the ``.xml`` files. For ``fnbr``, the connection to the database must be specified in a ``.toml`` file (Check [config.toml.template](tools/build-fnbr-dataset/config.toml.template) to see possible keys, including specification of which corpora should be used to train the model). Finally, source ``ehr`` requires both a database connection and the ``ehr_originaldb_path`` for the file containing the unanoymized version of the annotated sentences. 

Another essential parameter is ``--structure_db`` (default: ehr_db), which specifies whether ``fnbr`` or ``ehr`` should be the main FrameNet structure. All other annotations will be mapped to that of the structure DB, guaranteeing that no inconsistent FEs or Frames are used to train the model.

A typical command looks like this:

```shell
python build-fnbr-data.py data/ --sources=fn17,fnbr,ehr --db_config=./build-fnbr-dataset/config.toml --fn17_path=./fndata-1.7 --ehr_originaldb_path=./gbv-original-sents.csv --tokenizer=trankit
```

For a compreehensive explanation of all of its parameters run after installing required packages (use Python >=3.11):

```shell
python build-fnbr-data.py --help
```

### Running training procedure

When the docker image is built, it already configures relevant paths for the training procedure. To run the default training, two volumes must be mapped:

- **Data**: this is where the training data is located, the folder must contain the files ``train.jsonl``, ``dev.jsonl``, ``test.jsonl`` and ``ontology``;
- **Checkpoint**: this is the folder in the host machine where the model checkpoints will be saved (_There's no need to create this folder, only map it, the docker process will create it automatically_).

Suppose the **Data** and **Checkpoint** folders are ``data`` and ``model-checkpoint`` in the current folder. The run command should be:

```shell
sudo docker run -v $(pwd)/data/:/srv/data -v $(pwd)/model-checkpoint:/srv/checkpoint/ --gpus all -it "span-finder"
```

Once inside the container, you can run the following training command:

```shell
allennlp train -s $CHECKPOINT_PATH --include-package sftp config/fn.jsonnet
```

It is important to note that ``$CHECKPOINT_PATH`` is already set on the image. To check all the default configurations for paths, check [.env.default](.env.default). Just be careful: if any of those paths are changed, the mapping of volumes need to change as well. The most important value is ``CUDA``. By default, the process will try to use **cuda:0**. To train on CPU, use the following command when running the container:

```shell
sudo docker run -e CUDA=[-1] -v $(pwd)/data/:/srv/data -v $(pwd)/model-checkpoint:/srv/checkpoint/ -it "span-finder"
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

## Running demo

The docker image also contains the files required to run the demo. To do so, when running the container the folder with the ``model`` files must be mapped as volume to /srv/model. The ports also need to be mapped for that to work:

```shell
sudo docker run -v $(pwd)/model/:/srv/model -p 7749:7749 --gpus all --entrypoint python -it "span-finder" tools/demo/flask_backend.py
```

## Using batch inference (simple)

This repo also includes a script for easy batch inference. The script reads pure text data from ```STDIN``` and outputs one JSON for each line to ```STDOUT```. To use it, create a **txt** file having a single sentnece per line. Assuming that the file is ```sentences.txt```, use this command:

```shell
cat sentences.txt | sudo docker run -v $(pwd)/model/:/srv/model --gpus all --entrypoint python -i "span-finder" scripts/predict_span_batch.py
```

This will print out results. To save them, direct ```STDOUT``` to a **.jsonl** file (e.g. ```lome_outputs.jsonl```):

```shell
cat sentences.txt | sudo docker run -v $(pwd)/model/:/srv/model --gpus all --entrypoint python -i "span-finder" scripts/predict_span_batch.py > lome_outputs.jsonl
```
