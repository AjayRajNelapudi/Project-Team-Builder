import numpy as np
from sklearn.cluster import KMeans

class Ranker:
    def __init__(self, ranks=4):
        self.ranks = ranks

    def cluster(self, students):
        model = KMeans(n_clusters=self.ranks)
        model.fit(students)
        return model.cluster_centers_, model.labels_

    def weight(self, datapoint):
        feature_maxs = np.array([10, 22, 20, 5, 10, 10, 5, 10])
        feature_weights = np.array([0.2, 0.5, 0.5, 0.5, 0.3, 0.1, 0.2, 0.4])
        product_factor = feature_weights / feature_maxs
        datapoint_weight = np.sum(np.array(datapoint) * product_factor)
        return datapoint_weight

    def rank_clusters(self, cluster_centers):
        clusters = list(map(list, cluster_centers))
        clusters.sort(key=self.weight, reverse=True)
        return clusters

    def group_by_rank(self, students, cluster_centers, labels):
        ranks = {}
        for student_index in range(len(students)):
            label = labels[student_index]
            if label not in ranks:
                rank_info = {'cluster_center': cluster_centers[label], 'students': [student_index]}
                ranks[label] = rank_info
            else:
                ranks[label]['students'].append(student_index)

        return ranks

    def rank_students(self, designations, ranked_clusters):
        ranked_students = {}
        for designation_info in designations.values():
            cluster_center = list(designation_info['cluster_center'])
            rank = ranked_clusters.index(cluster_center) + 1
            ranked_students[rank] = designation_info

        return ranked_students

    def rank(self, students):
        cluster_centers, labels = self.cluster(students)
        ranks = self.group_by_rank(students, cluster_centers, labels)
        ranked_clusters = self.rank_clusters(cluster_centers)
        ranked_students = self.rank_students(ranks, ranked_clusters)
        return ranked_students

if __name__ == "__main__":
    import csv

    with open("dataset.csv") as datafile:
        dataset_reader = csv.reader(datafile, quoting=csv.QUOTE_NONNUMERIC)
        dataset = np.array(list(dataset_reader))

    student_ranker = Ranker()
    student_ranks = student_ranker.rank(dataset)

    groups = [[rank, student_info['students']] for rank, student_info in student_ranks.items()]
    groups.sort()

    for group in groups:
        print(group)

