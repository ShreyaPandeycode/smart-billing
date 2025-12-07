# tracker.py
import math

class SimpleTracker:
    def __init__(self, max_dist=90):
        self.next_id = 1
        self.tracks = {}  # id -> centroid (x,y)
        self.max_dist = max_dist

    def _centroid(self, bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)//2, (y1+y2)//2)

    def assign(self, detections):
        """
        detections: list of dicts with 'bbox'
        returns detections with 'track_id' assigned
        """
        assigned = []
        for d in detections:
            c = self._centroid(d['bbox'])
            best = None; best_dist = None
            for tid, tc in self.tracks.items():
                dist = math.hypot(tc[0]-c[0], tc[1]-c[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist; best = tid
            if best_dist is not None and best_dist < self.max_dist:
                self.tracks[best] = c
                d['track_id'] = best
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = c
                d['track_id'] = tid
            assigned.append(d)
        return assigned
