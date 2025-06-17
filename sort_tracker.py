from sort import KalmanBoxTracker
import numpy as np

class Sort:
    """
    Simple Online and Realtime Tracker (SORT)
    """

    def __init__(self, max_age=5, min_hits=3):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, dets=np.empty((0, 5))):
        """
        Updates trackers with new detections.
        """
        self.frame_count += 1
        new_trackers = []
        for t in self.trackers:
            pred = t.predict()
            if np.any(np.isnan(pred)):
                continue
            new_trackers.append(t)

        self.trackers = new_trackers

        for det in dets:
            tracker = KalmanBoxTracker(det)
            self.trackers.append(tracker)

        results = []
        for t in self.trackers:
            state = t.get_state()
            results.append([state[0], state[1], state[2], state[3], t.id])

        return np.array(results)
