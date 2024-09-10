import torch

def save_checkpoint(state, is_best, output_folder, ckpt_filename="last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save({
            'model_state_dict': state["model_state_dict"]
        }, f"{output_folder}/best_model.pth")


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