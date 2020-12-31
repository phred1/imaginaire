# DuckieTown Sim2real with Nvidia UNIT

To reproduce our results, you will need :
- A Simulator dataset and a Real dataset from Duckiebot logs. Both need to be of equal length.
- Nvidia Docker as well as Nvidia drivers installed. 

Then, make a train/test split with your dataset.
Create a folder `/dataset/sim2real_raw` which follows this directory structure
 ```
    /dataset/sim2real_raw
            - test
                - images_a:
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
                - images_b
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
            - train
                - images_a
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
                - images_b
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
 ```
Here, `images_a` and `images_b` correspond to the real and sim environment. Both are interchangeable
as long as `a` is always associated to the same environment and same thing for `b`. Make sure to have the files 
saved as `.jpeg` and not `.jpg`, as this will cause some issues. 

Once the raw data is set up, we need to build the lmdb dataset by running the following command:

`bash scripts/build_lmdb.sh unit sim2real`

Then we need to build and start the docker container that will allow us to run train our model with :

`bash scripts/build_docker.sh 20.05`

Followed by:

`bash start_local_docker.sh 20.05`

For more installation details, please refer to  [this README](https://github.com/phred1/imaginaire/blob/master/INSTALL.md)

Once the container is successfully started, run this command to train the model:

```
python -m torch.distributed.launch --nproc_per_node=1 inference.py \
--config configs/projects/unit/sim2real/sim2real.yaml \
--output_dir projects/unit/output/sim2real
``` 

To test the trained model, simply create a directory
`/configs/projects/unit/test_data` with the following structure: 
```
    - images_a:
        - 0002.jpeg
        - 0008.jpeg
        - 0033.jpeg
    - images_b
        - 0002.jpeg
        - 0008.jpeg
        - 0033.jpeg
```
To before testing the model on test data, we need to take the model checkpoint that was created in the `logs/**Date**_sim2real` folder. 

Then, copy the checkpoint to a new folder of your choice, and run the inference command:
```
python -m torch.distributed.launch --nproc_per_node=1 inference.py 
--config configs/projects/unit/simn2real/sim2real.yaml
--checkpoint checkpoint_path_you_choosed
--output_dir projects/unit/output/sim2real
```
The output dir should contain your generated data. To switch the domain translation from a -> b to b -> a, simply change the `a2b` field in `configs/projects/unit/sim2real/sim2real.yaml` to `False`