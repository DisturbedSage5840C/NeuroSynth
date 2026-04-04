variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "eks_cluster_name" {
  type    = string
  default = "neurosynth-eks"
}

variable "eks_cluster_role_arn" {
  type = string
}

variable "eks_node_role_arn" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "timescale_kms_key_arn" {
  type = string
}

variable "eks_oidc_provider_arn" {
  type = string
}

variable "audit_bucket_name" {
  type = string
}

variable "healthlake_kms_key_arn" {
  type = string
}
