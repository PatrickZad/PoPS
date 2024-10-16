import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog

from .cuhk_sysu import load_cuhk_sysu
from .prw import load_prw
from .movie_net import load_movie_net
from .coco_ch import load_coco_ch



# TODO change evaluator type to "query"
def register_cuhk_sysu_all(datadir):
    name = "CUHK-SYSU_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)

    name = "CUHK-SYSU_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")

    name = "CUHK-SYSU_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "CUHK-SYSU_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_movie_net_all(datadir):
    name = "MovieNet_" + "Train_app10"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app10"))
    MapperCatalog.register(name, mappers.MovieNetMapper)

    name = "MovieNet_" + "Train_app30"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app30"))
    MapperCatalog.register(name, mappers.MovieNetMapper)

    name = "MovieNet_" + "Train_app50"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app50"))
    MapperCatalog.register(name, mappers.MovieNetMapper)

    name = "MovieNet_" + "Train_app70"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app70"))
    MapperCatalog.register(name, mappers.MovieNetMapper)

    name = "MovieNet_" + "Train_app100"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app100"))
    MapperCatalog.register(name, mappers.MovieNetMapper)

    name = "MovieNet_" + "GalleryTestG2000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG2000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


    name = "MovieNet_" + "GalleryTestG4000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG4000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


    name = "MovieNet_" + "GalleryTestG10000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG10000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")

    name = "MovieNet_" + "TestG2000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "MovieNet_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "MovieNet_" + "TestG10000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG10000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_prw_all(datadir):
    name = "PRW_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwMapper)

    name = "PRW_Query"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

    name = "PRW_Gallery"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")




def register_cococh_all(datadir):
    name = "COCO-CH"
    DatasetCatalog.register(
        name, lambda: load_coco_ch(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)




_root = os.getenv("PS_DATASETS", "Data")
register_movie_net_all(os.path.join(_root, "movienet"))
register_cuhk_sysu_all(os.path.join(_root, "cuhk_sysu"))
register_prw_all(os.path.join(_root, "PRW"))
register_cococh_all(os.path.join(_root, "DetData"))
