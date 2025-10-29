# WandB Integration Guide for ANN Power System Classification

## What Was Added

Your ANN classification notebook has been enhanced with Weights & Biases (WandB) monitoring to track:

### 1. **Real-time Metrics**
   - Training loss per epoch
   - Test/validation loss per epoch
   - Test accuracy per epoch
   - Learning rate tracking

### 2. **Model Information**
   - Architecture visualization
   - Gradient tracking
   - Parameter histograms
   - Model checkpoints

### 3. **Visualizations**
   - Training loss curves
   - Confusion matrix (both static and interactive)
   - Classification metrics

### 4. **Hyperparameters Logged**
   - Learning rate: 0.001
   - Number of epochs: 50
   - Architecture: [128, 64, 32] hidden layers
   - Optimizer: Adam
   - Number of PMUs: 10
   - Features per PMU: 14
   - Number of classes: 5
   - Dataset split information

## Key Code Changes

### 1. Installation & Login
```python
!pip install wandb -qU
import wandb
wandb.login(key="your-api-key")
```

### 2. Initialize Run
```python
wandb.init(
    project="power-system-classification",
    name="ann-pmu-classification",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        # ... other hyperparameters
    }
)
```

### 3. Watch Model
```python
wandb.watch(model, log="all", log_freq=10)
```

### 4. Log Metrics in Training Loop
```python
wandb.log({
    "epoch": epoch + 1,
    "train_loss": loss.item(),
    "test_loss": test_loss.item(),
    "test_accuracy": test_accuracy,
    "learning_rate": optimizer.param_groups[0]['lr']
})
```

### 5. Log Visualizations
```python
wandb.log({"training_loss_plot": wandb.Image(plt)})
wandb.log({"confusion_matrix": wandb.Image(plt)})
```

### 6. Save Model
```python
torch.save(model_state, "model.pt")
wandb.save("model.pt")
```

### 7. Finish Run
```python
wandb.finish()
```

## How to Use

1. **Run the notebook**: Execute all cells in order
2. **View results**: After running, WandB will provide a URL link to your dashboard
3. **Explore dashboard**: 
   - View real-time training curves
   - Compare different runs
   - Download model checkpoints
   - Share results with your team

## WandB Dashboard Features

### Charts Tab
- Training/test loss over time
- Accuracy curves
- Custom visualizations

### System Tab
- GPU/CPU utilization
- Memory usage
- System metrics

### Model Tab
- Gradient histograms
- Parameter distributions
- Layer-wise statistics

### Artifacts Tab
- Saved model checkpoints
- Training data snapshots

### Logs Tab
- Console output
- Training progress

## Benefits of WandB Integration

1. **No More Manual Tracking**: Automatically logs all metrics
2. **Reproducibility**: All hyperparameters and results saved
3. **Comparison**: Easy to compare multiple training runs
4. **Collaboration**: Share dashboards with team members
5. **Visualization**: Interactive charts and plots
6. **Model Versioning**: Track different model versions
7. **Remote Monitoring**: Check training progress from anywhere

## Advanced Features (Optional)

### Set Up Alerts
```python
if accuracy < threshold:
    wandb.alert(
        title="Low Accuracy",
        text=f"Accuracy {accuracy} is below threshold"
    )
```

### Log Custom Tables
```python
table = wandb.Table(columns=["epoch", "loss", "accuracy"])
table.add_data(epoch, loss, acc)
wandb.log({"results": table})
```

### Sweep for Hyperparameter Tuning
```python
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [32, 64, 128]}
    }
}
sweep_id = wandb.sweep(sweep_config, project="your-project")
wandb.agent(sweep_id, function=train)
```

## Troubleshooting

### Issue: API Key Not Working
- Check if the key is correct
- Try interactive login: `wandb.login()` without key parameter

### Issue: No Data Showing
- Ensure `wandb.init()` is called before training
- Check that `wandb.log()` is inside the training loop
- Verify internet connection

### Issue: Too Much Data Logged
- Reduce logging frequency with `log_freq` parameter
- Use `wandb.watch(model, log_freq=100)` for less frequent updates

## Next Steps

1. **Run Multiple Experiments**: Try different hyperparameters
2. **Use Sweeps**: Automate hyperparameter search
3. **Team Collaboration**: Share your WandB project link
4. **Compare Models**: Use WandB Reports for comparisons
5. **Deploy Models**: Use WandB artifacts for model deployment

## Resources

- WandB Documentation: https://docs.wandb.ai/
- PyTorch Integration: https://docs.wandb.ai/guides/integrations/pytorch
- Example Projects: https://wandb.ai/wandb/examples

---

**Your WandB Project**: Once you run the notebook, look for the URL printed in the output. It will look like:
```
View run at https://wandb.ai/your-username/power-system-classification/runs/xyz123
```

Click this link to access your interactive dashboard!
