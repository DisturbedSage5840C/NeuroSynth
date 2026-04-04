resource "aws_eks_cluster" "this" {
  name     = "neurosynth-eks"
  role_arn = "arn:aws:iam::123456789012:role/neurosynth-eks-cluster"
  version  = "1.29"

  vpc_config {
    subnet_ids = ["subnet-aaaaaaaa", "subnet-bbbbbbbb"]
  }
}

resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "cpu-ng"
  node_role_arn   = "arn:aws:iam::123456789012:role/neurosynth-eks-node"
  subnet_ids      = ["subnet-aaaaaaaa", "subnet-bbbbbbbb"]
  instance_types  = ["m6i.4xlarge"]
  scaling_config { desired_size = 3, min_size = 1, max_size = 10 }
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "gpu-ng"
  node_role_arn   = "arn:aws:iam::123456789012:role/neurosynth-eks-node"
  subnet_ids      = ["subnet-aaaaaaaa", "subnet-bbbbbbbb"]
  instance_types  = ["g5.12xlarge"]
  scaling_config { desired_size = 2, min_size = 0, max_size = 8 }
}
