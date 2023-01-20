library(tidyverse)
library(SingleCellExperiment)
library(CytoTalk)
library(reticulate)


input_file <- "/tmp/pycharm_project/scheme/simulated_data/full_adata.h5ad"
output_dir <- "/tmp/cytotalk"

reticulate::use_condaenv("scheme_env", conda = "/opt/conda/bin/conda")
sc <- import("scanpy")
# Read the counts data
adata <- sc$read_h5ad(input_file)
lr_pairs <- data.frame(
    ligand=as.character(as.integer(adata$uns['ligands'])),
    receptor=as.character(as.integer(adata$uns['receptors']))
)
sce <- SingleCellExperiment(
    assays = list(counts = t(adata$X)),
    colData = adata$obs,
    rowData = adata$var
)
colnames(sce) <- as.character(colData(sce)$cell_type)
# Cytotalk requires logcounts
sce <- scuttle::logNormCounts(sce)

# Run Cytotalk
# Following https://github.com/tanlabcode/CytoTalk tutorial
lst_scrna <- CytoTalk::from_single_cell_experiment(sce)

celltypes <- unique(colData(sce)$cell_type)
celltype_pairs <- expand.grid(celltypes, celltypes, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)

# Run Cytotalk on all celltype pairs
for (i in 1:length(celltype_pairs)) {
  celltype1 <- celltype_pairs[i, 1]
  celltype2 <- celltype_pairs[i, 2]
  print(paste0("Running Cytotalk for celltype pair ", celltype1, " and ", celltype2))

  # FIXME
  results <- CytoTalk::run_cytotalk(
    lst_scrna,
    celltype1,
    celltype2,
    pcg = rownames(sce),  # All of our genes are "protein-coding"
    lrp = lr_pairs,
    dir_out = output_dir
  )
}