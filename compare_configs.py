"""
Compare different training configurations and provide recommendations
"""

import yaml
import argparse
from pathlib import Path

def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compare_configs(config1_path, config2_path):
    """Compare two configurations and highlight differences"""
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)
    
    print("\n" + "="*80)
    print("Configuration Comparison")
    print("="*80)
    print(f"\nConfig 1: {config1_path}")
    print(f"Config 2: {config2_path}\n")
    
    # Model comparison
    print("📦 MODEL CONFIGURATION")
    print("-" * 80)
    model1 = config1['model']
    model2 = config2['model']
    
    print(f"{'Parameter':<25} {'Config 1':<20} {'Config 2':<20} {'Change'}")
    print("-" * 80)
    print(f"{'Model Name':<25} {model1['name']:<20} {model2['name']:<20} {get_emoji(model1['name'], model2['name'])}")
    print(f"{'Freeze Backbone':<25} {str(model1['freeze_backbone']):<20} {str(model2['freeze_backbone']):<20} {get_emoji(model1['freeze_backbone'], model2['freeze_backbone'])}")
    print(f"{'Freeze Layers':<25} {model1['freeze_layers']:<20} {model2['freeze_layers']:<20} {get_emoji(model1['freeze_layers'], model2['freeze_layers'])}")
    
    # Training comparison
    print("\n🎓 TRAINING CONFIGURATION")
    print("-" * 80)
    train1 = config1['training']
    train2 = config2['training']
    
    print(f"{'Parameter':<25} {'Config 1':<20} {'Config 2':<20} {'Change'}")
    print("-" * 80)
    print(f"{'Epochs':<25} {train1['num_epochs']:<20} {train2['num_epochs']:<20} {get_emoji(train1['num_epochs'], train2['num_epochs'])}")
    print(f"{'Learning Rate':<25} {train1['learning_rate']:<20} {train2['learning_rate']:<20} {get_emoji_lr(train1['learning_rate'], train2['learning_rate'])}")
    print(f"{'Optimizer':<25} {train1['optimizer']:<20} {train2['optimizer']:<20} {get_emoji(train1['optimizer'], train2['optimizer'])}")
    print(f"{'Scheduler':<25} {train1['scheduler']:<20} {train2['scheduler']:<20} {get_emoji(train1['scheduler'], train2['scheduler'])}")
    print(f"{'Weight Decay':<25} {train1['weight_decay']:<20} {train2['weight_decay']:<20} {get_emoji(train1['weight_decay'], train2['weight_decay'])}")
    print(f"{'Early Stop Patience':<25} {train1['early_stopping_patience']:<20} {train2['early_stopping_patience']:<20} {get_emoji(train1['early_stopping_patience'], train2['early_stopping_patience'])}")
    
    warmup1 = train1.get('warmup_epochs', 0)
    warmup2 = train2.get('warmup_epochs', 0)
    print(f"{'Warmup Epochs':<25} {warmup1:<20} {warmup2:<20} {get_emoji(warmup1, warmup2)}")
    
    # Data comparison
    print("\n📊 DATA CONFIGURATION")
    print("-" * 80)
    data1 = config1['data']
    data2 = config2['data']
    
    print(f"{'Parameter':<25} {'Config 1':<20} {'Config 2':<20} {'Change'}")
    print("-" * 80)
    print(f"{'Batch Size':<25} {data1['batch_size']:<20} {data2['batch_size']:<20} {get_emoji(data1['batch_size'], data2['batch_size'])}")
    print(f"{'Num Workers':<25} {data1['num_workers']:<20} {data2['num_workers']:<20} {get_emoji(data1['num_workers'], data2['num_workers'])}")
    
    # Loss comparison
    print("\n📉 LOSS CONFIGURATION")
    print("-" * 80)
    loss1 = config1['loss']
    loss2 = config2['loss']
    
    print(f"{'Parameter':<25} {'Config 1':<20} {'Config 2':<20} {'Change'}")
    print("-" * 80)
    print(f"{'Loss Type':<25} {loss1['type']:<20} {loss2['type']:<20} {get_emoji(loss1['type'], loss2['type'])}")
    
    # Augmentation comparison
    print("\n🎨 AUGMENTATION CONFIGURATION")
    print("-" * 80)
    aug1 = config1['augmentation']
    aug2 = config2['augmentation']
    
    print(f"{'Parameter':<25} {'Config 1':<20} {'Config 2':<20} {'Change'}")
    print("-" * 80)
    print(f"{'Brightness':<25} {aug1['brightness']:<20} {aug2['brightness']:<20} {get_emoji(aug1['brightness'], aug2['brightness'])}")
    print(f"{'Contrast':<25} {aug1['contrast']:<20} {aug2['contrast']:<20} {get_emoji(aug1['contrast'], aug2['contrast'])}")
    print(f"{'Rotation':<25} {aug1['rotation']:<20} {aug2['rotation']:<20} {get_emoji(aug1['rotation'], aug2['rotation'])}")
    print(f"{'Random Crop':<25} {str(aug1.get('random_crop', False)):<20} {str(aug2.get('random_crop', False)):<20} {get_emoji(aug1.get('random_crop', False), aug2.get('random_crop', False))}")
    
    print("\n" + "="*80)
    print_recommendations(config1, config2)

def get_emoji(val1, val2):
    """Get emoji for comparison"""
    if val1 == val2:
        return "="
    else:
        return "✓" if is_better(val1, val2) else "⚠️"

def get_emoji_lr(lr1, lr2):
    """Special handling for learning rate"""
    if lr1 == lr2:
        return "="
    elif lr1 > 0.0005 and lr2 <= 0.0003:
        return "✓ Better"
    elif lr2 > 0.0005 and lr1 <= 0.0003:
        return "⚠️ Was better"
    else:
        return "Changed"

def is_better(val1, val2):
    """Determine if val2 is better than val1"""
    # This is a simplified heuristic
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return val2 > val1
    return True

def print_recommendations(config1, config2):
    """Print recommendations based on config comparison"""
    print("\n💡 RECOMMENDATIONS")
    print("="*80)
    
    train1 = config1['training']
    train2 = config2['training']
    
    recommendations = []
    
    # Learning rate check
    if train2['learning_rate'] < train1['learning_rate']:
        recommendations.append("✓ Lower learning rate should provide more stable training")
    
    # Optimizer check
    if train2['optimizer'] == 'adamw' and train1['optimizer'] != 'adamw':
        recommendations.append("✓ AdamW optimizer handles weight decay better than Adam")
    
    # Warmup check
    if train2.get('warmup_epochs', 0) > 0 and train1.get('warmup_epochs', 0) == 0:
        recommendations.append("✓ Learning rate warmup prevents early training instability")
    
    # Early stopping check
    if train2['early_stopping_patience'] > train1['early_stopping_patience']:
        recommendations.append("✓ Longer patience allows model to explore more before stopping")
    
    # Model check
    model_sizes = {'resnet18': 1, 'resnet34': 2, 'resnet50': 3, 'resnet101': 4}
    if model_sizes.get(config2['model']['name'], 0) > model_sizes.get(config1['model']['name'], 0):
        recommendations.append("✓ Larger model has more capacity to learn complex patterns")
    
    # Weight decay check
    if train2['weight_decay'] > train1['weight_decay']:
        recommendations.append("✓ Higher weight decay provides better regularization")
    
    # Loss function check
    if config2['loss']['type'] == 'huber' and config1['loss']['type'] == 'mse':
        recommendations.append("✓ Huber loss is more robust to outliers than MSE")
    
    if recommendations:
        print("\nPositive Changes:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\nNo significant improvements detected.")
    
    print("\n" + "="*80)

def analyze_single_config(config_path):
    """Analyze a single configuration and provide recommendations"""
    config = load_config(config_path)
    
    print("\n" + "="*80)
    print(f"Configuration Analysis: {config_path}")
    print("="*80)
    
    issues = []
    warnings = []
    good_practices = []
    
    # Check learning rate
    lr = config['training']['learning_rate']
    if lr > 0.001:
        issues.append(f"⚠️  Learning rate {lr} is very high - may cause unstable training")
    elif lr > 0.0005:
        warnings.append(f"⚠️  Learning rate {lr} might be too high for fine-tuning")
    else:
        good_practices.append(f"✓ Learning rate {lr} is appropriate for fine-tuning")
    
    # Check model
    model_name = config['model']['name']
    if model_name == 'resnet18':
        warnings.append(f"⚠️  ResNet18 has limited capacity - consider ResNet34 or ResNet50")
    else:
        good_practices.append(f"✓ {model_name} is a good choice for this task")
    
    # Check warmup
    warmup = config['training'].get('warmup_epochs', 0)
    if warmup == 0:
        warnings.append("⚠️  No learning rate warmup - may cause early instability")
    else:
        good_practices.append(f"✓ Using {warmup} warmup epochs")
    
    # Check optimizer
    optimizer = config['training']['optimizer']
    if optimizer == 'adamw':
        good_practices.append("✓ AdamW is recommended for fine-tuning")
    elif optimizer == 'adam':
        warnings.append("⚠️  Consider using AdamW instead of Adam")
    
    # Check early stopping
    patience = config['training']['early_stopping_patience']
    if patience < 10:
        warnings.append(f"⚠️  Early stopping patience {patience} might be too aggressive")
    else:
        good_practices.append(f"✓ Early stopping patience {patience} is reasonable")
    
    # Check batch size
    batch_size = config['data']['batch_size']
    if batch_size < 16:
        warnings.append(f"⚠️  Batch size {batch_size} is small - gradients may be noisy")
    else:
        good_practices.append(f"✓ Batch size {batch_size} is good")
    
    # Print results
    if issues:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if good_practices:
        print("\n✅ GOOD PRACTICES:")
        for practice in good_practices:
            print(f"  {practice}")
    
    # Overall score
    total_checks = len(issues) + len(warnings) + len(good_practices)
    score = (len(good_practices) / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"\n📊 Configuration Score: {score:.1f}/100")
    
    if score >= 80:
        print("   Rating: ⭐⭐⭐ Excellent")
    elif score >= 60:
        print("   Rating: ⭐⭐ Good")
    elif score >= 40:
        print("   Rating: ⭐ Fair")
    else:
        print("   Rating: ⚠️  Needs Improvement")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Compare training configurations')
    parser.add_argument('--config1', type=str, help='First configuration file')
    parser.add_argument('--config2', type=str, help='Second configuration file')
    parser.add_argument('--analyze', type=str, help='Analyze a single configuration')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_single_config(args.analyze)
    elif args.config1 and args.config2:
        compare_configs(args.config1, args.config2)
    else:
        # Default comparison
        print("Comparing demo config with improved config...\n")
        if Path('config_demo.yaml').exists() and Path('config_improved.yaml').exists():
            compare_configs('config_demo.yaml', 'config_improved.yaml')
        else:
            print("Error: config_demo.yaml or config_improved.yaml not found")
            print("\nUsage:")
            print("  python compare_configs.py --config1 config1.yaml --config2 config2.yaml")
            print("  python compare_configs.py --analyze config.yaml")

if __name__ == '__main__':
    main()

