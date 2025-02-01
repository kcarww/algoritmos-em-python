from typing import List
from data_point import DataPoint
from kmeans import KMeans
import matplotlib.pyplot as plt

class Album(DataPoint):
    def __init__(self, name: str, year: int, length: float, tracks: int):
        super().__init__([length, tracks])
        self.name = name
        self.year = year
        self.length = length
        self.tracks = tracks

    def __repr__(self):
        return f"{self.name} ({self.year}): {self.length} minutes"
    
if __name__ == '__main__':
    albums: List[Album] = [
        Album("Piper at the Gates of Dawn", 1967, 41.47, 9),
        Album("A Saucerful of Secrets", 1968, 39.23, 7),
        Album("More", 1969, 44.27, 13),
        Album("Ummagumma", 1969, 69.51, 12),
        Album("Atom Heart Mother", 1970, 52.06, 5),
        Album("Meddle", 1971, 46.46, 6),
        Album("Obscured by Clouds", 1972, 40.25, 10),
        Album("The Dark Side of the Moon", 1973, 42.49, 9),
        Album("Wish You Were Here", 1975, 44.28, 5),
        Album("Animals", 1977, 41.42, 5),
        Album("The Wall", 1979, 81.9, 26),
        Album("The Final Cut", 1983, 43.0, 13),
        Album("A Momentary Lapse of Reason", 1987, 51.14, 11),
        Album("The Division Bell", 1994, 66.32, 11),
        Album("The Endless River", 2014, 53.27, 21)
    ]
    
    kmeans: KMeans[Album] = KMeans(2, albums)
    clusters: List[KMeans.Cluster] = kmeans.run()
    for index, cluster in enumerate(clusters):
        print(f"Cluster {index} avg length: {cluster.centroid.dimensions[0]} avg tracks {cluster.centroid.dimensions[1]}: {cluster.points}\n")