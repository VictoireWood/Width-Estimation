import torch
import logging
import shutil
from collections import OrderedDict


def save_checkpoint(state, is_best, is_best_recall_25, is_best_recall_50, output_folder, ckpt_filename="last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save({
            'model_state_dict': state["model_state_dict"]
        }, f"{output_folder}/best_model_loss.pth")

    if is_best_recall_25:
        torch.save({
            'model_state_dict': state["model_state_dict"]
        }, f"{output_folder}/best_model_recall_25.pth")

    if is_best_recall_50:
        torch.save({
            'model_state_dict': state["model_state_dict"]
        }, f"{output_folder}/best_model_recall_50.pth")


def resume_model(model: torch.nn.Module, resume_info = {}):
    logging.info(f"Resuming model from {resume_info['resume_model']}")
    checkpoint = torch.load(resume_info['resume_model'])
    model_state_dict = checkpoint['model_state_dict']
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    device = resume_info['device']
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    return model

def resume_train_with_params(output_folder, model, model_optimizer, resume_train_info = {}):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {resume_train_info['resume_train']}")
    checkpoint = torch.load(resume_train_info['resume_train'])
    start_epoch_num = checkpoint['epoch_num']
    model_state_dict = checkpoint['model_state_dict']
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    model.load_state_dict(model_state_dict)

    model = model.to(resume_train_info['device'])
    model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_train_loss = checkpoint['best_train_loss']


    # Copy best model to current output_folder
    shutil.copy(resume_train_info['resume_train'].replace("last_checkpoint.pth", "best_model_loss.pth"), output_folder)
    shutil.copy(resume_train_info['resume_train'].replace("last_checkpoint.pth", "best_model_recall_25.pth"), output_folder)
    shutil.copy(resume_train_info['resume_train'].replace("last_checkpoint.pth", "best_model_recall_50.pth"), output_folder)
    return model, model_optimizer, best_train_loss, start_epoch_num
    

# def resume_model(args, model, classifiers):
#     logging.info(f"Resuming model from {args.resume_model}")
#     checkpoint = torch.load(args.resume_model)

#     model_state_dict = checkpoint["model_state_dict"]
#     if list(model_state_dict.keys())[0].startswith('module'):
#         model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
#     model.load_state_dict(model_state_dict)

#     assert len(classifiers) == len(checkpoint["classifiers_state_dict"]), \
#         f"{len(classifiers)}, {len(checkpoint['classifiers_state_dict'])}"

#     for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
#         # Move classifiers to GPU before loading their optimizers
#         c = c.to(args.device)
#         c.load_state_dict(sd)
#         c = c.cpu()

#     return model, classifiers


# def move_to_device(optimizer, device):
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if torch.is_tensor(v):
#                 state[k] = v.to(device)