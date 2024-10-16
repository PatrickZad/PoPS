from .mapper import SearchMapper


class PrwMapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
