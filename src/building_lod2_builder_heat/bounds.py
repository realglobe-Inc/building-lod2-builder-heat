from pyproj import CRS, Transformer


class Bounds:
    """
    境界のインデクスを表すクラス
    """

    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __str__(self):
        return f"{self.ltrb}"

    def __repr__(self):
        return f"Bounds{self.ltrb}"

    @property
    def ltrb(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.top - self.bottom


class GeoBounds:
    """
    境界の座標を表すクラス
    """

    def __init__(self, left: float, top: float, right: float, bottom: float, crs: CRS):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.crs = crs

    def __str__(self):
        return f"{(self.left, self.top, self.right, self.bottom, self.crs.to_string())}"

    def __repr__(self):
        return f"GeoBounds{str(self)}"

    def transform_to(self, target_crs: CRS) -> "GeoBounds":
        """
        座標系を変換した新しい境界を返します。

        :param target_crs: 変換先の座標系
        :type target_crs: CRS
        :returns: 変換後の新しいBoundsオブジェクト
        :rtype: GeoBounds
        """
        if self.crs == target_crs:
            return self
        transformer = Transformer.from_crs(self.crs, target_crs, always_xy=True)
        left, top = transformer.transform(self.left, self.top)
        right, bottom = transformer.transform(self.right, self.bottom)
        return GeoBounds(left, top, right, bottom, target_crs)

    @property
    def ltrb(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom
