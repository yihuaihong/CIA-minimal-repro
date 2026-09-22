# import os
#
#
# def init_wandb_training(training_args):
#     """
#     Helper function for setting up Weights & Biases logging tools.
#     """
#     if training_args.wandb_entity is not None:
#         os.environ["WANDB_ENTITY"] = training_args.wandb_entity
#     if training_args.wandb_project is not None:
#         os.environ["WANDB_PROJECT"] = training_args.wandb_project
#     if training_args.wandb_run_group is not None:
#         os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group


import os
# offline mode
# def init_wandb_training(training_args):
#     """
#     Helper function for setting up Weights & Biases logging tools.
#     """
#
#     # ✅ 离线模式（不连接 wandb.cloud，只写本地）
#     os.environ["WANDB_MODE"] = "offline"
#     os.environ["WANDB_DIR"] = "/gpfsnyu${SCRATCH}/wandb"  # 修改为你想保存日志的路径
#     os.environ["WANDB__SERVICE_WAIT"] = "300"  # 防止超时
#
#     # ✅ 基本配置：项目名、用户、组等
#     if getattr(training_args, "wandb_entity", None):
#         os.environ["WANDB_ENTITY"] = training_args.wandb_entity
#     else:
#         os.environ["WANDB_ENTITY"] = "local"
#
#     if getattr(training_args, "wandb_project", None):
#         os.environ["WANDB_PROJECT"] = training_args.wandb_project
#     else:
#         os.environ["WANDB_PROJECT"] = "gemma-sft"
#
#     if getattr(training_args, "wandb_run_group", None):
#         os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
#
#     # ✅ 防止 Hugging Face Trainer 尝试上传模型
#     os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
#     os.environ["HF_HUB_OFFLINE"] = "1"


# online mode
def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """

    # ✅ 在线模式（连接 wandb.cloud）
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_DIR"] = "${HOME}/Research_Projects/Other_Code/open_r1/wandb"  # 缓存目录可以保留，用于临时存储
    # os.environ["WANDB__SERVICE_WAIT"] = "300"  # 在线模式通常不需要强制等待，除非网络极差

    # ✅ 基本配置：项目名、用户、组等
    if getattr(training_args, "wandb_entity", None):
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    else:
        # os.environ["WANDB_ENTITY"] = "local"
        # 在线模式下，如果没有指定 entity，最好移除这一行，
        # 让 wandb 默认使用你本地 `wandb login` 时的用户名。
        pass

    if getattr(training_args, "wandb_project", None):
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    else:
        os.environ["WANDB_PROJECT"] = "gemma-sft"

    if getattr(training_args, "wandb_run_name", None):
        os.environ["WANDB_RUN_NAME"] = training_args.wandb_run_name

    if getattr(training_args, "wandb_run_group", None):
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group

    # ✅ 如果你是“在线”模式，通常也需要连接 HF Hub，建议注释掉下面这两行
    # os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # os.environ["HF_HUB_OFFLINE"] = "1"
