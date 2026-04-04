terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.45" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.30" }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project            = "NeuroSynth"
      Environment        = "prod"
      DataClassification = "PHI"
    }
  }
}

module "networking" {
  source = "./modules/networking"
}

module "storage" {
  source = "./modules/storage"
}

module "iam" {
  source = "./modules/iam"
  eks_oidc_provider_arn = var.eks_oidc_provider_arn
}

module "eks" {
  source = "./modules/eks"
  cluster_role_arn = var.eks_cluster_role_arn
  node_role_arn    = var.eks_node_role_arn
  subnet_ids       = var.private_subnet_ids
  cluster_name     = var.eks_cluster_name
}

module "databases" {
  source = "./modules/databases"
  timescale_kms_key_arn = var.timescale_kms_key_arn
}

module "monitoring" {
  source = "./modules/monitoring"
  audit_bucket_name = var.audit_bucket_name
  healthlake_kms_key_arn = var.healthlake_kms_key_arn
}
