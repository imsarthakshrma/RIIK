"""
Ablation Study Framework for Kolosis
Systematically test each component's contribution to performance.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path

from neural_networks.autograd.transformer_torch import GPT as BaselineGPT
from neural_networks.kolosis import KolosisTransformer
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import TinyStoriesDataset

class AblationStudy:
    """Framework for running ablation studies"""
    
    def __init__(self, output_dir='experiments/ablation_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def train_model(self, model, train_loader, val_loader, epochs, lr, model_name, device='cuda'):
        """
        Train a model and track metrics.
        
        Returns:
            results: Dict with training metrics
        """
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        results = {
            'model_name': model_name,
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'total_time': 0,
            'final_train_loss': 0,
            'final_val_loss': 0,
            'convergence_epoch': None,  # Epoch where val loss < threshold
        }
        
        start_time = time.time()
        best_val_loss = float('inf')
        convergence_threshold = 2.0  # Define convergence threshold
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            results['train_loss'].append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            results['val_loss'].append(avg_val_loss)
            
            # Track convergence
            if avg_val_loss < convergence_threshold and results['convergence_epoch'] is None:
                results['convergence_epoch'] = epoch
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            epoch_time = time.time() - epoch_start
            results['epoch_times'].append(epoch_time)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")
        
        results['total_time'] = time.time() - start_time
        results['final_train_loss'] = results['train_loss'][-1]
        results['final_val_loss'] = results['val_loss'][-1]
        results['best_val_loss'] = best_val_loss
        
        print(f"\nTraining complete!")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Final val loss: {results['final_val_loss']:.4f}")
        print(f"Convergence epoch: {results['convergence_epoch'] if results['convergence_epoch'] else 'Not reached'}")
        
        return results
    
    def run_baseline_gpt(self, train_loader, val_loader, config, device='cuda'):
        """Train baseline GPT"""
        model = BaselineGPT(
            vocab_size=config['vocab_size'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            block_size=config['block_size'],
            dropout=config['dropout']
        )
        
        results = self.train_model(
            model, train_loader, val_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            model_name='Baseline GPT',
            device=device
        )
        
        self.results['baseline_gpt'] = results
        return results
    
    def run_kolosis_full(self, train_loader, val_loader, config, device='cuda'):
        """Train full Kolosis"""
        model = KolosisTransformer(
            vocab_size=config['vocab_size'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            block_size=config['block_size'],
            dropout=config['dropout']
        )
        
        results = self.train_model(
            model, train_loader, val_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            model_name='Kolosis (Full)',
            device=device
        )
        
        self.results['kolosis_full'] = results
        return results
    
    def save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / 'ablation_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    def generate_report(self):
        """Generate markdown report"""
        report = "# Kolosis Ablation Study Results\n\n"
        
        if 'baseline_gpt' not in self.results or 'kolosis_full' not in self.results:
            report += "Incomplete results. Run both baseline and Kolosis experiments.\n"
            return report
        
        baseline = self.results['baseline_gpt']
        kolosis = self.results['kolosis_full']
        
        report += "## Summary\n\n"
        report += "| Metric | Baseline GPT | Kolosis | Improvement |\n"
        report += "|--------|--------------|---------|-------------|\n"
        
        # Final validation loss
        val_improvement = (baseline['final_val_loss'] - kolosis['final_val_loss']) / baseline['final_val_loss'] * 100
        report += f"| Final Val Loss | {baseline['final_val_loss']:.4f} | {kolosis['final_val_loss']:.4f} | {val_improvement:+.1f}% |\n"
        
        # Convergence speed
        baseline_conv = baseline['convergence_epoch'] if baseline['convergence_epoch'] else 'N/A'
        kolosis_conv = kolosis['convergence_epoch'] if kolosis['convergence_epoch'] else 'N/A'
        
        if isinstance(baseline_conv, int) and isinstance(kolosis_conv, int):
            conv_improvement = (baseline_conv - kolosis_conv) / baseline_conv * 100
            report += f"| Convergence Epoch | {baseline_conv} | {kolosis_conv} | {conv_improvement:+.1f}% |\n"
        else:
            report += f"| Convergence Epoch | {baseline_conv} | {kolosis_conv} | N/A |\n"
        
        # Training time
        time_overhead = (kolosis['total_time'] - baseline['total_time']) / baseline['total_time'] * 100
        report += f"| Total Time (s) | {baseline['total_time']:.1f} | {kolosis['total_time']:.1f} | {time_overhead:+.1f}% |\n"
        
        report += "\n## Detailed Results\n\n"
        report += "### Baseline GPT\n"
        report += f"- Final train loss: {baseline['final_train_loss']:.4f}\n"
        report += f"- Final val loss: {baseline['final_val_loss']:.4f}\n"
        report += f"- Best val loss: {baseline['best_val_loss']:.4f}\n"
        report += f"- Total time: {baseline['total_time']:.2f}s\n\n"
        
        report += "### Kolosis (Full)\n"
        report += f"- Final train loss: {kolosis['final_train_loss']:.4f}\n"
        report += f"- Final val loss: {kolosis['final_val_loss']:.4f}\n"
        report += f"- Best val loss: {kolosis['best_val_loss']:.4f}\n"
        report += f"- Total time: {kolosis['total_time']:.2f}s\n\n"
        
        # Save report
        report_file = self.output_dir / 'ablation_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_file}")
        return report


if __name__ == "__main__":
    # Quick test of ablation framework
    print("Ablation Study Framework - Test Mode")
    
    # Small config for testing
    config = {
        'vocab_size': 50,
        'n_embd': 32,
        'n_head': 2,
        'n_layer': 1,
        'block_size': 16,
        'dropout': 0.1,
        'epochs': 5,
        'lr': 0.001,
        'batch_size': 4,
    }
    
    print(f"Config: {config}")
    print("Framework ready for experiments!")
