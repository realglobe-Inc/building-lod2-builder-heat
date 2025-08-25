from pyproj import CRS, Transformer
from shapely.geometry.polygon import Polygon


class GeoOutline:
    """
    外形線を表すクラス
    """

    def __init__(self, polygon: Polygon, crs: CRS):
        self.polygon = polygon
        self.crs = crs

    def transform_to(self, target_crs: CRS) -> "GeoOutline":
        """
        座標系を変換した新しい外形線を返します。

        :param target_crs: 変換先の座標系
        :type target_crs: CRS
        :returns: 変換後の新しいGeoOutlineオブジェクト
        :rtype: GeoOutline
        """
        if self.crs == target_crs:
            return self
        transformer = Transformer.from_crs(self.crs, target_crs, always_xy=True)
        transformed_coords = [
            transformer.transform(x, y) for x, y in self.polygon.exterior.coords
        ]
        transformed_polygon = Polygon(transformed_coords)
        return GeoOutline(transformed_polygon, target_crs)
