import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to

class SparseAutoencoder(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps, type="residual"):
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        self.use_unlabeled_y = config.use_unlabeled_y # Expect x,y,m from unlabeled loaders and train on the unlabeled y
        self.recon_alpha = config.recon_alpha
        self.type = config.type

        self.logged_fields.append('recon_loss')
        self.logged_fields.append('class_loss')

        if self.type == "sparse_recon":

            self.sparse_alpha = config.sparse_alpha
            self.logged_fields.append('sparse_loss')

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        autoencoder, outputs = self.get_model_output(x, y_true)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'representation': autoencoder,
            'input': x
        }
        if unlabeled_batch is not None:
            if self.use_unlabeled_y: # expect loaders to return x,y,m
                x, y, metadata = unlabeled_batch
                y = move_to(y, self.device)
            else:
                x, metadata = unlabeled_batch    
            x = move_to(x, self.device)
            results['unlabeled_metadata'] = metadata
            if self.use_unlabeled_y:
                results['unlabeled_y_pred'] = self.get_model_output(x, y)
                results['unlabeled_y_true'] = y
            results['unlabeled_g'] = self.grouper.metadata_to_group(metadata).to(self.device)
        return results

    def objective(self, results):
        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        if self.type == "sparse_residual":
            recon_loss = torch.nn.MSELoss()(results['input'], results['representation'] + results['input'])
        elif self.type == "sparse_recon":
            recon_loss = torch.nn.MSELoss()(results['input'], results['representation'])
            sparse_loss = torch.abs(results['representation']).mean()
            self.save_metric_for_logging(results, 'sparse_loss', sparse_loss)
        elif self.type == "disentangled":
            recon_loss = torch.nn.MSELoss()(results['input'], results['representation'])
            #disentangle_loss = de_loss(intermediate_output, labels, domains)

        self.save_metric_for_logging(results, 'recon_loss', recon_loss)
        self.save_metric_for_logging(results, 'class_loss', labeled_loss)


        if self.use_unlabeled_y and 'unlabeled_y_true' in results:
            unlabeled_loss = self.loss.compute(
                results['unlabeled_y_pred'], 
                results['unlabeled_y_true'], 
                return_dict=False
            )
            lab_size = len(results['y_pred'])
            unl_size = len(results['unlabeled_y_pred'])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (lab_size + unl_size)
        else:

            if self.type == "residual":
                return self.recon_alpha * recon_loss + labeled_loss
            elif self.type == "sparse_recon":
                return labeled_loss + self.recon_alpha * recon_loss + self.sparse_alpha * sparse_loss
