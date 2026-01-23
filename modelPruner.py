import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Optional, Dict
import copy


class ModelPruner:
    """
    A flexible pruning system for PyTorch models that supports multiple pruning strategies.
    
    Args:
        model: PyTorch model to prune
        param_names: List of parameter names to prune (e.g., ['conv1.weight', 'fc.weight'])
        strategy: Pruning strategy to use
        prune_ratio: Percentage of parameters to prune (0.0 to 1.0)
        **kwargs: Additional strategy-specific parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        param_names: List[str],
        strategy: str,
        prune_ratio: float = 0.5,
        **kwargs
    ):
        self.model = model
        self.param_names = param_names
        self.strategy = strategy.lower()
        self.prune_ratio = prune_ratio
        self.kwargs = kwargs
        
        # Store original parameters for potential restoration
        self.original_params = {}
        self._save_original_params()
        
    def _save_original_params(self):
        """Save original parameters before pruning"""
        for name in self.param_names:
            param = self._get_parameter(name)
            if param is not None:
                self.original_params[name] = param.data.clone()
    
    def _get_parameter(self, param_name: str) -> Optional[torch.nn.Parameter]:
        """Get parameter by name from model"""
        try:
            # Navigate nested attributes
            attrs = param_name.split('.')
            obj = self.model
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            print(f"Warning: Parameter {param_name} not found in model")
            return None
    
    def _set_parameter(self, param_name: str, value: torch.Tensor):
        """Set parameter value by name"""
        attrs = param_name.split('.')
        obj = self.model
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], nn.Parameter(value))
    
    def prune(self) -> Dict[str, float]:
        """
        Apply pruning based on the selected strategy.
        
        Returns:
            Dictionary containing pruning statistics for each parameter
        """
        stats = {}
        
        if self.strategy == "magnitude":
            stats = self._magnitude_pruning()
        elif self.strategy == "random":
            stats = self._random_pruning()
        elif self.strategy == "l1_structured":
            stats = self._l1_structured_pruning()
        elif self.strategy == "l2_structured":
            stats = self._l2_structured_pruning()
        elif self.strategy == "gradient":
            stats = self._gradient_based_pruning()
        elif self.strategy == "percentile":
            stats = self._percentile_pruning()
        elif self.strategy == "topk":
            stats = self._topk_pruning()
        elif self.strategy == "movement":
            stats = self._movement_pruning()
        elif self.strategy == "variance":
            stats = self._variance_based_pruning()
        elif self.strategy == "structured_channel":
            stats = self._structured_channel_pruning()
        elif self.strategy == "structured_filter":
            stats = self._structured_filter_pruning()
        elif self.strategy == "global_magnitude":
            stats = self._global_magnitude_pruning()
        else:
            raise ValueError(f"Unknown pruning strategy: {self.strategy}")
        
        return stats
    
    def _magnitude_pruning(self) -> Dict[str, float]:
        """Prune weights with smallest absolute magnitude"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # Calculate threshold based on magnitude
            weights = param.data.abs().flatten()
            threshold = torch.quantile(weights, self.prune_ratio)
            
            # Create mask
            mask = (param.data.abs() > threshold).float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _random_pruning(self) -> Dict[str, float]:
        """Randomly prune parameters"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # Create random mask
            mask = torch.rand_like(param.data) > self.prune_ratio
            mask = mask.float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _l1_structured_pruning(self) -> Dict[str, float]:
        """Structured pruning based on L1 norm of filters/neurons"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # Assume first dimension is output channels/filters
            if len(param.shape) < 2:
                print(f"Warning: {param_name} has insufficient dimensions for structured pruning")
                continue
            
            # Calculate L1 norm per output channel
            l1_norms = param.data.abs().view(param.shape[0], -1).sum(dim=1)
            threshold = torch.quantile(l1_norms, self.prune_ratio)
            
            # Create mask for entire filters
            mask = (l1_norms > threshold).float()
            
            # Expand mask to match parameter shape
            for _ in range(len(param.shape) - 1):
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(param.data)
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_filters = param.shape[0]
            pruned_filters = (l1_norms <= threshold).sum().item()
            stats[param_name] = pruned_filters / total_filters
        
        return stats
    
    def _l2_structured_pruning(self) -> Dict[str, float]:
        """Structured pruning based on L2 norm of filters/neurons"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            if len(param.shape) < 2:
                print(f"Warning: {param_name} has insufficient dimensions for structured pruning")
                continue
            
            # Calculate L2 norm per output channel
            l2_norms = param.data.view(param.shape[0], -1).norm(dim=1, p=2)
            threshold = torch.quantile(l2_norms, self.prune_ratio)
            
            # Create mask for entire filters
            mask = (l2_norms > threshold).float()
            
            # Expand mask to match parameter shape
            for _ in range(len(param.shape) - 1):
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(param.data)
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_filters = param.shape[0]
            pruned_filters = (l2_norms <= threshold).sum().item()
            stats[param_name] = pruned_filters / total_filters
        
        return stats
    
    def _gradient_based_pruning(self) -> Dict[str, float]:
        """Prune based on gradient information (requires gradients)"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            if param.grad is None:
                print(f"Warning: {param_name} has no gradient. Run backward pass first.")
                continue
            
            # Use gradient magnitude as importance score
            importance = (param.data.abs() * param.grad.abs()).flatten()
            threshold = torch.quantile(importance, self.prune_ratio)
            
            # Create mask
            importance_reshaped = param.data.abs() * param.grad.abs()
            mask = (importance_reshaped > threshold).float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _percentile_pruning(self) -> Dict[str, float]:
        """Prune based on percentile threshold"""
        stats = {}
        percentile = self.kwargs.get('percentile', self.prune_ratio * 100)
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # Calculate threshold
            weights_abs = param.data.abs().flatten()
            threshold = torch.quantile(weights_abs, percentile / 100.0)
            
            # Create mask
            mask = (param.data.abs() > threshold).float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _topk_pruning(self) -> Dict[str, float]:
        """Keep only top-k largest magnitude weights"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # Calculate number of weights to keep
            total_params = param.numel()
            k = int(total_params * (1 - self.prune_ratio))
            
            # Find top-k weights
            weights_flat = param.data.abs().flatten()
            topk_values, topk_indices = torch.topk(weights_flat, k)
            
            # Create mask
            mask = torch.zeros_like(weights_flat)
            mask[topk_indices] = 1.0
            mask = mask.reshape(param.shape)
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            pruned_params = total_params - k
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _movement_pruning(self) -> Dict[str, float]:
        """
        Movement pruning: prune weights that didn't change much during training
        Requires storing initial parameters
        """
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            if param_name not in self.original_params:
                print(f"Warning: No original parameters stored for {param_name}")
                continue
            
            # Calculate movement (change from original)
            original = self.original_params[param_name]
            movement = (param.data - original).abs().flatten()
            threshold = torch.quantile(movement, self.prune_ratio)
            
            # Create mask based on movement
            movement_reshaped = (param.data - original).abs()
            mask = (movement_reshaped > threshold).float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _variance_based_pruning(self) -> Dict[str, float]:
        """Prune based on variance of weights in local neighborhoods"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # For simplicity, use global variance as importance
            # In practice, could use local variance in kernels
            weights = param.data.flatten()
            variance = weights.var()
            
            # Prune weights close to mean (low contribution to variance)
            mean = weights.mean()
            deviation = (weights - mean).abs()
            threshold = torch.quantile(deviation, self.prune_ratio)
            
            # Create mask
            deviation_reshaped = (param.data - mean).abs()
            mask = (deviation_reshaped > threshold).float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def _structured_channel_pruning(self) -> Dict[str, float]:
        """Prune entire input channels"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # For conv layers: [out_channels, in_channels, H, W]
            if len(param.shape) < 2:
                continue
            
            # Calculate importance per input channel (dim=1)
            importance = param.data.abs().sum(dim=0)  # Sum over output channels
            while len(importance.shape) > 1:
                importance = importance.sum(dim=-1)  # Sum over spatial dimensions
            
            threshold = torch.quantile(importance, self.prune_ratio)
            
            # Create mask for channels
            mask = (importance > threshold).float()
            
            # Expand mask to match parameter shape
            mask = mask.view(1, -1, *([1] * (len(param.shape) - 2)))
            mask = mask.expand_as(param.data)
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_channels = importance.numel()
            pruned_channels = (importance <= threshold).sum().item()
            stats[param_name] = pruned_channels / total_channels
        
        return stats
    
    def _structured_filter_pruning(self) -> Dict[str, float]:
        """Prune entire output filters"""
        stats = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # For conv/linear layers: first dim is output features
            if len(param.shape) < 2:
                continue
            
            # Calculate importance per output filter (dim=0)
            importance = param.data.abs().view(param.shape[0], -1).sum(dim=1)
            threshold = torch.quantile(importance, self.prune_ratio)
            
            # Create mask for filters
            mask = (importance > threshold).float()
            
            # Expand mask to match parameter shape
            for _ in range(len(param.shape) - 1):
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(param.data)
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_filters = importance.numel()
            pruned_filters = (importance <= threshold).sum().item()
            stats[param_name] = pruned_filters / total_filters
        
        return stats
    
    def _global_magnitude_pruning(self) -> Dict[str, float]:
        """Prune globally across all specified parameters"""
        stats = {}
        
        # Collect all weights
        all_weights = []
        param_shapes = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            all_weights.append(param.data.abs().flatten())
            param_shapes[param_name] = param.shape
        
        # Concatenate all weights
        global_weights = torch.cat(all_weights)
        
        # Calculate global threshold
        threshold = torch.quantile(global_weights, self.prune_ratio)
        
        # Apply pruning to each parameter
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            # Create mask
            mask = (param.data.abs() > threshold).float()
            
            # Apply mask
            param.data *= mask
            
            # Calculate statistics
            total_params = param.numel()
            pruned_params = (mask == 0).sum().item()
            stats[param_name] = pruned_params / total_params
        
        return stats
    
    def restore_original(self):
        """Restore original parameters before pruning"""
        for param_name, original_value in self.original_params.items():
            param = self._get_parameter(param_name)
            if param is not None:
                param.data.copy_(original_value)
    
    def get_sparsity(self) -> Dict[str, float]:
        """Calculate current sparsity for each parameter"""
        sparsity = {}
        
        for param_name in self.param_names:
            param = self._get_parameter(param_name)
            if param is None:
                continue
            
            total_params = param.numel()
            zero_params = (param.data == 0).sum().item()
            sparsity[param_name] = zero_params / total_params
        
        return sparsity
    
    def print_stats(self, stats: Dict[str, float]):
        """Print pruning statistics"""
        print("\n" + "="*60)
        print(f"Pruning Strategy: {self.strategy}")
        print(f"Target Prune Ratio: {self.prune_ratio:.2%}")
        print("="*60)
        total_params = 0
        pruned_params = 0
        for param_name, ratio in stats.items():
            param = self._get_parameter(param_name)
            total = param.numel() if param is not None else 0
            pruned = int(total * ratio)
            pruned_params += pruned
            total_params += total
            print(f"{param_name:30s} | Pruned: {pruned:8d}/{total:8d} ({ratio:.2%})")
        
        print("="*60 + "\n")
        return pruned_params, total_params


# Example usage and demonstration
if __name__ == "__main__":
    from Generator import *
    import pickle
    from util import set_seed
    SAME_DATA_GT = False
    SAME_DATA_BIAS = False
    SAME_DATA_SEEN = False
    SAME_NETWORK_INIT = False
    # Create a simple model for demonstration
    device = torch.device('cuda:0')
    #load generator, load dataset, load weights
    main_folder = "./zpruneAnalysisSaves/"
    folders = os.listdir(main_folder)
    all_results_all_files = {}
    for f in folders:
        folder = os.path.join(main_folder,f)+"/"
        data_name = "data_0.npz"
        one_hot_data_name = "one_hot_data_0.npz"
        weights_name = "weights_0_.npz"
        embed_name = "embedding_dict_0_.pikl"
        if f not in all_results_all_files:
            all_results_all_files[f] = {}
        
        all_results = all_results_all_files[f]
        data = np.load(folder + data_name)
        one_hot_data = np.load(folder + one_hot_data_name)
        x = data['x']
        y = data['y']
        one_hot_x = torch.tensor(one_hot_data['x'],dtype=torch.float32).to(device)
        with open(folder + embed_name, 'rb') as file:
            embed_dict = pickle.load(file)
        weights = np.load(folder+weights_name)['w'].flatten()

        checkpoint = torch.load(folder + "/generator_checkpoint.pt", map_location="cpu")
        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]
        fixed_seed = np.random.randint(1e6)
        all_rngs = []
        print('SETTING SEED: ', fixed_seed)
        for _ in range(1):
            all_rngs.append(set_seed(fixed_seed,
                                    device,
                                data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                data_gen =SAME_DATA_SEEN,
                                network_init=SAME_NETWORK_INIT,))
            
        generator = DeepSetNet(rngs=all_rngs[0],
                                num_features = config["num_features"],
                                layers = config["layers"],
                                sample_size = config["sample_size"],
                                dropout = config["dropout"],
                                batch_size = config["batch_size"],
                                temperature = config["temperature"],
                                embedding_dict=config['embedding_dict']).to(device)
        missing, unexpected = generator.load_state_dict(checkpoint["state_dict"], strict=False)


        param_names = [name for name, param in generator.named_parameters() 
                    if param.requires_grad and 'weight' in name]
        generator.set_eval()
        cur_weights = generator.get_weights(one_hot_x)
        
        # Define parameters to prune
        #param_names = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
        
        # Test different pruning strategies
        '''
        strategies = [
            'magnitude',
            'random',
            'l1_structured',
            'l2_structured',
            'topk',
            'global_magnitude'
        ]'''
        strategies = [
            "magnitude",
            "random",
            "l1_structured",
            "l2_structured",
            "gradient",
            "percentile",
            "topk",
            "movement",
            "variance",
            "structured_channel",
            "structured_filter",
            "global_magnitude",
            ]
        
        #print("Testing different pruning strategies:\n")
        
        
        all_results['no_pruning'] = {}
        all_results['no_pruning'][0] = [(cur_weights @ y).item(), 1, 1]
        for strategy in strategies:
            # Create fresh model for each strategy
            missing, unexpected = generator.load_state_dict(checkpoint["state_dict"], strict=False)
            test_model = generator
            if strategy not in all_results:
                all_results[strategy] = {}
            for pr in [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2]:
                if pr not in all_results[strategy]:
                    all_results[strategy][pr] = 0
                # Create pruner
                pruner = ModelPruner(
                    model=test_model,
                    param_names=param_names,
                    strategy=strategy,
                    prune_ratio=pr 
                )
                
                # Apply pruning
                stats = pruner.prune()
                # Print statistics
                pruned_params, total_params = pruner.print_stats(stats)
                
                if False:
                    # Show actual sparsity
                    sparsity = pruner.get_sparsity()
                    print(f"Actual sparsity achieved:")
                    for name, sparse_ratio in sparsity.items():
                        print(f"  {name}: {sparse_ratio:.2%}")
                    print()

                cur_weights = test_model.get_weights(one_hot_x)
                all_results[strategy][pr] = [(cur_weights @ y).item(),pruned_params, total_params]

    with open("pruning_results.pikl", 'wb') as file:
        pickle.dump(all_results_all_files, file)