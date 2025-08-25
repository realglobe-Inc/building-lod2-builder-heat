from pathlib import Path

from shapely import Polygon, unary_union

from building_lod2_builder_heat.commands.infer.outline import GeoOutline
from building_lod2_builder_heat.commands.infer.parameter import load_obj_crs_from_json


class Obj3D:
    """
    OBJファイルの3Dメッシュモデルを表すクラス
    """

    def __init__(self, vertices, raw_faces):
        """
        Obj3Dオブジェクトを初期化します。

        :param vertices: 頂点座標のリスト
        :type vertices: list
        :param raw_faces: 面の頂点インデックスのリスト
        :type raw_faces: list
        """
        self.vertices = vertices
        self.raw_faces = raw_faces
        self.faces = []
        for face_indices in raw_faces:
            face_vertices = [vertices[i] for i in face_indices]
            self.faces.append(face_vertices)

    @classmethod
    def load(cls, file_path):
        """
        OBJファイルから読み込んだObj3Dを返す

        :param file_path: OBJファイルのパス
        :type file_path: Path
        :returns: Obj3Dオブジェクト
        :rtype: Obj3D
        """
        vertices = []
        raw_faces = []

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("v "):
                    # 頂点座標の解析
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
                elif line.startswith("f "):
                    # 面の解析
                    parts = line.split()
                    face_indices = []
                    for part in parts[1:]:
                        # OBJファイルのインデックスは1から始まるため、0から始まるように調整
                        index = int(part.split("/")[0]) - 1
                        face_indices.append(index)
                    raw_faces.append(face_indices)

        return cls(vertices, raw_faces)

    def calculate_horizontal_outline(self) -> Polygon | None:
        """
        3Dメッシュモデルから水平面上の外形線ポリゴンを作成します。

        :returns: 水平面上の外形線（作成に失敗した場合はNone）
        :rtype: Polygon | None
        """
        # 各面を水平面に投影してポリゴンのリストを作成
        horizontal_polygons = []

        for face in self.faces:
            # 水平面(Z=0)に投影（X, Y座標のみ使用）
            poly = Polygon([(x, y) for x, y, _ in face])
            if poly.is_valid and poly.area > 0:
                horizontal_polygons.append(poly)

        if not horizontal_polygons:
            return None

        # 複数のポリゴンを結合して外形線を作成
        outline_polygon = unary_union(horizontal_polygons)

        # MultiPolygonの場合は最大面積のポリゴンを使用
        if hasattr(outline_polygon, "geoms"):
            outline_polygon = max(outline_polygon.geoms, key=lambda p: p.area)

        return outline_polygon.simplify(1e-8)


def load_outline_from_obj(obj_file_path: Path) -> GeoOutline | None:
    """
    OBJファイルから3Dメッシュモデルの水平面上の外形線を読み取る。

    :param obj_file_path: 3Dメッシュモデルを表すOBJファイルのパス
    :type obj_file_path: Path
    :returns: 3Dメッシュモデルの外形線と座標系
    :rtype: GeoOutline | None
    """
    outline = Obj3D.load(obj_file_path).calculate_horizontal_outline()
    if outline is None:
        return None

    # OBJファイルの拡張子をjsonにしたファイルから座標系を読み取る
    json_file = obj_file_path.with_suffix(".json")
    crs = load_obj_crs_from_json(json_file)
    if crs is None:
        print(f"警告: {json_file}から座標系を読み取れませんでした。")
        return None

    return GeoOutline(outline, crs)
