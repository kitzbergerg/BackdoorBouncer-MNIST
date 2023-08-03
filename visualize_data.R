setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(reticulate)
library(mclust)
library(ggplot2)

py_run_string("import pickle")

unpickle <- function(file_name) {
  py_run_string(paste0("with open('", file_name, "', 'rb') as f: data = pickle.load(f)"))
  py$`data`
}

data <- unpickle("data/feed_forward_output.pkl")

uuids <- sapply(data, function(x) x[[1]][[1]])
labels <- sapply(data, function(x) x[[2]][[1]])
second_to_last_layer_outputs <- t(sapply(data, function(x) x[[3]]))





library(Rtsne)
tsne_results <- Rtsne(second_to_last_layer_outputs)

visualization_data <- data.frame(
  X = tsne_results$Y[,1],
  Y = tsne_results$Y[,2],
  Labels = as.factor(labels)
)

library(ggplot2)
ggplot(visualization_data, aes(x = X, y = Y, color = Labels)) +
  geom_point() +
  theme_minimal() +
  labs(title = "t-SNE Visualization of Second-to-Last Layer Outputs",
       x = "t-SNE Dimension 1",
       y = "t-SNE Dimension 2") +
  scale_color_brewer(palette = "Set1")





library(mclust)
clustering_results <- Mclust(second_to_last_layer_outputs)

cluster_labels <- clustering_results$classification
model_params <- summary(clustering_results, parameters = TRUE)$parameters

pca_results <- prcomp(second_to_last_layer_outputs)

visualization_data <- data.frame(
  X = pca_results$x[,1],
  Y = pca_results$x[,2],
  Clusters = as.factor(cluster_labels),
  Labels = as.factor(labels)
)

library(ggplot2)
ggplot(visualization_data, aes(x = X, y = Y, color = Labels)) +
  geom_point() +
  theme_minimal() +
  labs(title = "Cluster Visualization of Second-to-Last Layer Outputs",
       x = "PCA Dimension 1",
       y = "PCA Dimension 2") +
  scale_color_brewer(palette = "Set1")
