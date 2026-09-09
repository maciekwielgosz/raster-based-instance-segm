# ==============================================================================
# ETAP C: Segmentacja Drzew (MCWS na CHM)
# ==============================================================================
options(encoding = "UTF-8")

library(lidR)
library(terra)
library(fs)
library(ForestTools)
library(sf)
library(future)
library(furrr)
library(progressr)

t_start <- Sys.time()

# 1. Ścieżki i foldery
# FOLDER WEJSCIOWY - chunki z CHM:
input_chm_dir <- "data_input"

# FOLDER WYNIKOWY:
output_dir <- "data_output"

dirs <- list(
  Segmentation = file.path(output_dir, "Segmentation3")
)
lapply(dirs, fs::dir_create)

# 2. Parametry
PROJECT_CRS <- "EPSG:2180"
n_cores     <- 1
plan(multisession, workers = n_cores)

# Wyczytywanie plikow wsadowych z input_chm_dir
chm_list <- list.files(input_chm_dir, pattern = "^chm_.*\\.tif$", full.names = TRUE)
chm_vrt_path <- file.path(input_chm_dir, "00_CHM_Full.vrt")

if(length(chm_list) == 0) stop(paste("Błąd: Brak plików CHM. Sprawdź folder:", input_chm_dir))

# 3. Segmentacja
message(">>> C1. Start Segmentacji...")

# funkcja Naslunda do szacowania dbh
dbh_naslund <- function(H, a = 10, b = 0.6) {
  Hm <- pmax(H - 1.3, 0); A <- -Hm * b; C <- -Hm * a; disc <- A^2 - 4 * C
  return((-A + sqrt(disc)) / 2)
}

# Sprawdzenie po ID, czy dany chunk nie został już wcześniej przetworzony
process_seg_chunk <- function(chm_file) {
  base_name <- fs::path_ext_remove(fs::path_file(chm_file))
  chunk_id <- sub("^chm_", "", base_name)

  out_crowns <- file.path(dirs$Segmentation, paste0("crowns_", chunk_id, ".gpkg"))
  out_ttops  <- file.path(dirs$Segmentation, paste0("ttops_", chunk_id, ".gpkg"))
  if (file.exists(out_crowns)) return(data.frame(ID = chunk_id, trees_count = NA))

  tryCatch({
    # Przygotowanie chunkow z buforem=25m (Edge Effect Mitigation)
    chm_r <- terra::rast(chm_file)
    core_ext <- terra::ext(chm_r)
    ext_buffer <- terra::ext(core_ext[1] - 25, core_ext[2] + 25, core_ext[3] - 25, core_ext[4] + 25)

    # wycinamy kafelek + jego sąsiedztwo z globalnego VRT
    local_vrt <- terra::rast(chm_vrt_path)
    chm_chunk <- terra::crop(local_vrt, ext_buffer)

    # Jeśli chunk jest pusty lub najwyższe drzewo ma poniżej 2m - POMIJAMY
    if (is.null(chm_chunk) || terra::global(chm_chunk, "max", na.rm = TRUE)[1,1] < 2) return(NULL)

    # Wygladzenie CHM filtrem medianowym (usuwanie artefaktow, zapobieganie wykrywaniu "falszywych" wierzcholkow na jednej koronie)
    chm_spikefree <- terra::focal(chm_chunk, w = 3, fun = "median", na.rm = TRUE)

    # Detekcja wierzchołków (Local Maximum Filter - LMF)
    f_ws <- function(z) ifelse(z < 10, 2, ifelse(z < 25, z * 0.1 + 0.3, z * 0.15 + 1))
    ttops <- lidR::locate_trees(chm_spikefree, lidR::lmf(ws = f_ws, hmin = 2))
    if (nrow(ttops) == 0) return(NULL)

    ttops_sf <- sf::st_as_sf(ttops)
    sf::st_crs(ttops_sf) <- PROJECT_CRS

    # docinanie wierzcholkow do granic kafla 500x500m
    coords <- sf::st_coordinates(ttops_sf)
    inside_core <- coords[,1] >= core_ext[1] & coords[,1] < core_ext[2] &
      coords[,2] >= core_ext[3] & coords[,2] < core_ext[4]

    ttops_final <- ttops_sf[inside_core, ]
    if (nrow(ttops_final) == 0) return(NULL)

    # Szacowanie piersnicy drzew (DBH)
    ttops_final$dbh <- round(dbh_naslund(ttops_final$Z), 2)

    # Segmentacja koron (MCWS - Marker-Controlled Watershed)
    crowns <- ForestTools::mcws(treetops = ttops, CHM = chm_spikefree, minHeight = 2, format = "polygons")
    crowns_sf <- sf::st_as_sf(crowns)
    sf::st_crs(crowns_sf) <- PROJECT_CRS


    # Filtracja koron - zostaja te, ktorych wierzcholki znajduja sie w chunku 500x500m
    crowns_final <- crowns_sf[crowns_sf$treeID %in% ttops_final$treeID, ]
    crowns_final$area_m2 <- as.numeric(sf::st_area(crowns_final))

    # Zapis
    sf::st_write(ttops_final, out_ttops, quiet = TRUE, append = FALSE, delete_dsn = TRUE)
    sf::st_write(crowns_final, out_crowns, quiet = TRUE, append = FALSE, delete_dsn = TRUE)

    return(data.frame(ID = chunk_id, trees_count = nrow(ttops_final)))
  }, error = function(e) { return(NULL) })
}

# Zrownoleglenie iteracji i podsumowanie wynikow
handlers(global = TRUE)
with_progress({
  processing_summary <- future_map(chm_list, process_seg_chunk, .progress = TRUE)
})

valid_summaries <- processing_summary[!sapply(processing_summary, is.null)]
if (length(valid_summaries) > 0) {
  summary_df <- do.call(rbind, valid_summaries)
  message("Wykryto drzew: ", sum(summary_df$trees_count, na.rm = TRUE))
}

message("Koniec Etapu C. Czas: ", round(difftime(Sys.time(), t_start, units = "mins"), 2), " min")