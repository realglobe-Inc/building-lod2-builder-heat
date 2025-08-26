from shapely import Polygon, unary_union


class Obj3D:
    """
    OBJファイルの3Dメッシュモデルを表すクラス
    """

    def __init__(
        self, vertices: list[tuple[float, float, float]], raw_faces: list[list[int]]
    ):
        """
        Obj3Dオブジェクトを初期化します。

        :param vertices: 頂点座標のリスト
        :param raw_faces: 面の頂点インデックスのリスト
        """
        self._vertices = vertices
        self._raw_faces = raw_faces
        self._faces = []
        for face_indices in raw_faces:
            face_vertices = [vertices[i] for i in face_indices]
            self._faces.append(face_vertices)

    @classmethod
    def load(cls, file_path):
        """
        OBJファイルから読み込んだObj3Dを返す

        :param file_path: OBJファイルのパス
        :return: Obj3Dオブジェクト
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

    def calculate_horizontal_outline(self) -> Polygon:
        """
        3Dメッシュモデルから水平面上の外形線ポリゴンを作成します。

        :return: 水平面上の外形線（作成に失敗した場合はNone）
        """
        # 各面を水平面に投影してポリゴンのリストを作成
        horizontal_polygons = []

        for face in self._faces:
            # 水平面(Z=0)に投影（X, Y座標のみ使用）
            poly = Polygon([(x, y) for x, y, _ in face])
            if poly.is_valid and poly.area > 0:
                horizontal_polygons.append(poly)

        # 複数のポリゴンを結合して外形線を作成
        outline_polygon = unary_union(horizontal_polygons)

        # MultiPolygonの場合は最大面積のポリゴンを使用
        if hasattr(outline_polygon, "geoms"):
            outline_polygon = max(outline_polygon.geoms, key=lambda p: p.area)

        return outline_polygon.simplify(1e-8)
