job_config = {}


def add_config(
    name,
    train_task_name,
    agg_name,
    train_dataset_name,
    train_collate_fn_name=None,
    test_task_name="TestTask",
    news_dataset_name="NewsDataset",
    user_dataset_name="UserDataset",
):
    job_config[name] = {}
    job_config[name]["train_task_name"] = train_task_name
    job_config[name]["agg_name"] = agg_name
    job_config[name]["train_dataset_name"] = train_dataset_name
    job_config[name]["train_collate_fn_name"] = train_collate_fn_name
    job_config[name]["test_task_name"] = test_task_name
    job_config[name]["news_dataset_name"] = news_dataset_name
    job_config[name]["user_dataset_name"] = user_dataset_name


add_config(
    name="Efficient-FedRec-Fast",
    train_task_name="BaseTrainTask",
    agg_name="BaseAggregator",
    train_dataset_name="TrainBaseDataset",
    train_collate_fn_name="train_base_collate_fn",
)
add_config(
    name="Efficient-FedRec",
    train_task_name="UserTrainTask",
    agg_name="UserAggregator",
    train_dataset_name="TrainUserDataset",
    train_collate_fn_name="train_user_collate_fn",
)
add_config(
    name="Efficient-FedRec-MPC",
    train_task_name="MPCTrainTask",
    agg_name="MPCAggregator",
    train_dataset_name="TrainMPCDataset",
    train_collate_fn_name="train_user_collate_fn",
)



if __name__ == "__main__":
    print(job_config)
