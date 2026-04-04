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
}

module "eks" {
  source = "./modules/eks"
}

module "databases" {
  source = "./modules/databases"
}

module "monitoring" {
  source = "./modules/monitoring"
}
