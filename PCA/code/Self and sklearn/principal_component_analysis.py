from collections.abc import Iterable
import csv

DataSet = list["Vector"]

class Vector(list[float]):
	def __init__(
		self,
		data_or_iterable: list[float] | Iterable[float] | None = None,
		*,
		size: int | None = None
	) -> None:
		if data_or_iterable is not None and size is not None:
			raise ValueError("cannot specify both size and data")
		elif size is not None:
			self.extend([0.0] * sizex)
		elif isinstance(data_or_iterable, list):
			self.extend(data_or_iterable)
		else:
			super().__init__(data_or_iterable or [])

	def dot(self, other: "Vector") -> float:
		if len(self) != len(other):
			raise ValueError(f"vectors must be of equal lengths: {self} {other}")
		return sum(
			a * b
			for a, b
			in zip(self, other)
		)

	def __add__(self, other: "Vector") -> "Vector":
		v = Vector(self)
		v += other
		return v

	def __iadd__(self, other: "Vector") -> "Vector":
		for i, value in enumerate(other):
			self[i] += value
		return self

	def __sub__(self, other: "Vector") -> "Vector":
		v = Vector(self)
		v -= other
		return v

	def __isub__(self, other: "Vector") -> "Vector":
		for i, value in enumerate(other):
			self[i] -= value
		return self

	def __truediv__(self, value: float) -> "Vector":
		v = Vector(self)
		v /= value
		return v

	def __itruediv__(self, value: float) -> "Vector":
		for i in range(len(self)):
			self[i] /= value
		return self

	def __str__(self) -> str:
		return ", ".join(
			f"{v:>6.2f}"
			for v
			in self
		)

def center_dataset(dataset: DataSet) -> DataSet:
	sum_vector = Vector(size=len(dataset[0]))
	for row in dataset:
		sum_vector += row
	average_vector = sum_vector/len(dataset)

	return [v - average_vector for v in dataset]

def calculate_covariance(dataset: DataSet, i: int, j: int) -> float:
	covariance = 0.0
	for row in dataset:
		covariance += row[i] * row[j]
	covariance /= len(dataset)
	return covariance

def find_eigenvectors(dataset: DataSet) -> tuple[list[float], list[Vector]]:
	from numpy.linalg import eig
	values, vectors = eig(dataset)

	return list(values), [Vector(v) for v in vectors]

def principal_component_analysis(raw_dataset: DataSet, keep: int = 1) -> DataSet:
	dataset = center_dataset(raw_dataset)

	dimensions = len(dataset[0])
	covariance_matrix = list(Vector(size=dimensions) for _ in range(dimensions))
	print(covariance_matrix)
	for i in range(dimensions):
		for j in range(i, dimensions):
			covariance_matrix[i][j] = calculate_covariance(dataset, i, j)
			if i != j:
				covariance_matrix[j][i] = covariance_matrix[i][j]

	eigenvalues, eigenvectors = find_eigenvectors(covariance_matrix)
	sorted_eigenvectors = sorted(
		zip(eigenvalues, eigenvectors),
		key=lambda e: e[0],
		reverse=True
	)

	components = sorted_eigenvectors[:keep]
	new_dataset: DataSet = []
	for p in dataset:
		new_dataset.append(Vector(p.dot(c) for _, c in components))

	return new_dataset

def main() -> None:
	dataset: DataSet = []
	with open("./dataset/data.csv", encoding="utf-8", newline="") as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)
		for r in reader:
			dataset.append(Vector(map(float, r)))

	print("original dataset:")
	for row in dataset:
		print(str(row))
	print()

	new_dataset = principal_component_analysis(dataset, keep=2)
	print("new dataset:")
	for row in new_dataset:
		print(str(row))


if __name__ == "__main__":
	main()
