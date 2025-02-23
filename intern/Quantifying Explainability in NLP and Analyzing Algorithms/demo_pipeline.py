#============= demo_pipeline.py =============
from data_loader import load_mimic_data
from model_builder import create_model
from mimic_utils import split_data, create_data_pipeline, evaluate_model
from mindspore import context, Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore import nn
def main():
    # 环境配置
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",
        max_call_depth=10000
    )
    
    # 数据加载
    print("\nLoading data...")
    X, y = load_mimic_data()
    
    # 数据划分
    print("\nSplitting data...")
    train_ds, test_ds = split_data(X, y)
    
    # 创建模型
    print(f"\nCreating model (input_dim={X.shape[1]})...")
    model = create_model(X.shape[1])
    
    # 训练配置
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
    loss_fn = nn.BCELoss()
    train_net = Model(model, loss_fn, optimizer, metrics={"acc": nn.Accuracy()})
    
    # 训练执行
    print("\nStarting training...")
    train_net.train(
        epoch=5,
        train_dataset=create_data_pipeline(train_ds, batch_size=64),
        callbacks=[LossMonitor(100), TimeMonitor()],
        dataset_sink_mode=False
    )
    
    # 评估
    print("\nEvaluating model...")
    test_ds = create_data_pipeline(test_ds, batch_size=128)
    metrics = evaluate_model(model, test_ds)
    
    print("\nFinal Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Average Probability: {metrics['avg_prob']:.4f}")  # 保持键名一致

if __name__ == "__main__":
    main()