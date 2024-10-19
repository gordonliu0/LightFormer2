
# If continuing from checkpoint:
if CONTINUE_TRAINING:
    EPOCHS=32
    checkpoint = torch.load("/Users/gordonliu/Documents/ml_projects/LightForker-2/src/checkpoints/final.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    last_epoch = 33
    scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_steps=0, warmup_start_factor=1, warmup_end_factor=1, T_0=32, T_mult=2)
    writer = SummaryWriter('/Users/gordonliu/Documents/ml_projects/LightForker-2/src/runs/Oct16_21-07-40_2021mbp.lan')

def continue_training():
    global_step = [772]
    for epoch in range(last_epoch+1, last_epoch+1+EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}, lr={current_lr}")
        writer.add_scalar("LR", current_lr, epoch)
        train(train_dataloader, model, loss_fn, optimizer, global_step, batches_per_log=10)
        _, _, metric, _, _, _ = validate(val_dataloader, model, val_loss_fn, epoch)
        scheduler.step()
        checkpointer.save_checkpoint(model, epoch=epoch, metric=metric)
    print(f"🎉 Finished training {EPOCHS} epochs for LightFormer2!")
    torch.save(model.state_dict(), "checkpoints/final.pth")
