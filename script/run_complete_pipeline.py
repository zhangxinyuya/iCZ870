#!/usr/bin/env python3
"""
Complete Core SNPs phylogenetic tree pipeline - Docker Version (V2)
✅ 使用Docker容器替代Conda环境
✅ 完全适配NCBI Datasets下载格式 (GCF_xxx.fna)
✅ 自动从.fna文件header提取菌株名，保留完整物种名
✅ 完整的ANI筛选 + SNP分析 + 建树流程

Author: Adapted for C. glutamicum analysis
Date: 2024-12
Modified: V2 - 增加完整物种名标签
"""

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from Bio import SeqIO
import shutil
import logging
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class SNPTreePipelineDocker:
    """完整的SNP系统发育树构建流程 - Docker版本"""

    def __init__(self,
                 genome_dir: str,
                 output_dir: str,
                 metadata_file: Optional[str] = None,
                 threads: int = 8,
                 ani_threshold: float = 99.9,
                 species_name: str = "Corynebacterium glutamicum",
                 docker_fastani: str = "staphb/fastani:latest",
                 docker_snippy: str = "staphb/snippy:latest",
                 docker_iqtree: str = "staphb/iqtree:latest"):
        """
        参数:
        genome_dir: 基因组文件夹（genomes/目录，包含.fna文件）
        output_dir: 输出目录
        metadata_file: 可选的元数据CSV文件
        threads: 线程数
        ani_threshold: ANI相似度阈值（>=此值视为冗余）
        species_name: 默认物种名（用于标签）
        docker_fastani: FastANI Docker镜像
        docker_snippy: Snippy Docker镜像
        docker_iqtree: IQ-TREE Docker镜像
        """
        self.genome_dir = Path(genome_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.metadata_file = metadata_file
        self.threads = threads
        self.ani_threshold = ani_threshold
        self.species_name = species_name

        self.docker_fastani = docker_fastani
        self.docker_snippy = docker_snippy
        self.docker_iqtree = docker_iqtree

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_genomes_dir = self.output_dir / "filtered_genomes"
        self.filtered_genomes_dir.mkdir(exist_ok=True)
        self.ani_results_dir = self.output_dir / "ani_analysis"
        self.ani_results_dir.mkdir(exist_ok=True)

        self.strain_to_full_label: Dict[str, str] = {}

        self._setup_logging()
        self._verify_docker()
        self.metadata_df = self._load_metadata()

    def _verify_docker(self):
        """验证Docker是否可用"""
        self.logger.info("\n验证Docker环境...")

        try:
            result = subprocess.run(["docker", "--version"],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Docker未安装或无法运行")

            self.logger.info(f"  ✅ {result.stdout.strip()}")

            images_to_check = [
                (self.docker_fastani, "FastANI"),
                (self.docker_snippy, "Snippy"),
                (self.docker_iqtree, "IQ-TREE"),
            ]

            for image, name in images_to_check:
                result = subprocess.run(
                    ["docker", "image", "inspect", image],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    self.logger.info(f"  ✅ {name} 镜像已存在: {image}")
                else:
                    self.logger.warning(f"  ⚠️  {name} 镜像未找到: {image}")
                    self.logger.warning(f"     将在首次使用时自动下载")

        except Exception as e:
            self.logger.error(f"❌ Docker验证失败: {e}")
            self.logger.error("\n请确保:")
            self.logger.error("  1. Docker已安装")
            self.logger.error("  2. Docker服务正在运行")
            self.logger.error("  3. 当前用户有权限运行Docker")
            sys.exit(1)

    def _get_docker_mount_path(self, local_path: Path) -> Tuple[str, str]:
        """获取Docker挂载路径"""
        local_abs = local_path.resolve()
        return str(local_abs), str(local_abs)

    def _run_docker(self, image: str, cmd: str, description: str,
                    mount_dirs: Optional[List[Path]] = None) -> Optional[str]:
        """在Docker容器中运行命令"""
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"Step: {description}")
        self.logger.info(f"Docker Image: {image}")
        self.logger.info(f"{'=' * 70}\n")

        mount_args = []
        if mount_dirs:
            for dir_path in mount_dirs:
                local_path, container_path = self._get_docker_mount_path(dir_path)
                mount_args.extend(["-v", f"{local_path}:{container_path}"])

        output_local, output_container = self._get_docker_mount_path(self.output_dir)
        mount_args.extend(["-v", f"{output_local}:{output_container}"])

        genome_local, genome_container = self._get_docker_mount_path(self.genome_dir)
        mount_args.extend(["-v", f"{genome_local}:{genome_container}"])

        docker_cmd = [
                         "docker", "run", "--rm",
                         "-u", f"{os.getuid()}:{os.getgid()}",
                     ] + mount_args + [image, "bash", "-c", cmd]

        self.logger.debug(f"Docker command: {' '.join(docker_cmd)}")

        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"❌ ERROR: {description} failed!")
            self.logger.error(f"STDERR: {result.stderr}")
            self.logger.error(f"STDOUT: {result.stdout}")
            return None

        self.logger.info(f"✅ {description} completed")
        if result.stdout:
            self.logger.debug(f"Output: {result.stdout[:500]}")

        return result.stdout

    def _setup_logging(self):
        """设置日志"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")

    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """加载元数据文件"""
        if not self.metadata_file:
            possible_paths = [
                self.genome_dir.parent / "metadata" / "genomes_summary.csv",
                Path("cglutamicum_pangenome/metadata/genomes_summary.csv"),
                self.output_dir / "genomes_summary.csv"
            ]

            for path in possible_paths:
                if path.exists():
                    self.metadata_file = str(path)
                    break

        if self.metadata_file and Path(self.metadata_file).exists():
            self.logger.info(f"加载元数据: {self.metadata_file}")
            df = pd.read_csv(self.metadata_file)
            self.logger.info(f"  找到 {len(df)} 个基因组的元数据")
            return df
        else:
            self.logger.warning("未找到元数据文件，将从.fna header解析菌株名")
            return None

    def parse_fna_for_species_and_strain(self, fna_file: Path) -> Tuple[str, str]:
        """
        从.fna文件的第一行header解析物种名和菌株名
        返回: (物种名, 菌株号)
        """
        try:
            with open(fna_file, 'r') as f:
                first_line = f.readline().strip()

            if not first_line.startswith('>'):
                return self.species_name, fna_file.stem

            header = first_line[1:]
            parts = header.split(' ', 1)
            if len(parts) < 2:
                return self.species_name, fna_file.stem

            desc = parts[1]
            words = desc.split()
            species = self.species_name
            strain = ""

            if len(words) >= 2 and words[0][0].isupper():
                if words[1][0].islower():
                    species = f"{words[0]} {words[1]}"
                    remaining = ' '.join(words[2:])
                    remaining = re.sub(r',?\s*(complete\s+)?(sequence|genome|chromosome).*$',
                                       '', remaining, flags=re.IGNORECASE)
                    remaining = re.sub(r'\s*,\s*$', '', remaining)

                    strain_match = re.search(r'(ATCC\s*\d+|strain\s+\S+|str\.\s*\S+|\S+)',
                                             remaining, flags=re.IGNORECASE)
                    if strain_match:
                        strain = strain_match.group(1).strip()
                        strain = re.sub(r'ATCC\s*', 'ATCC ', strain, flags=re.IGNORECASE)

            if not strain:
                filename = fna_file.stem
                atcc_match = re.search(r'(ATCC[_\s]*\d+)', filename, flags=re.IGNORECASE)
                if atcc_match:
                    strain = re.sub(r'ATCC[_\s]*', 'ATCC ', atcc_match.group(1))
                else:
                    strain = filename

            return species, strain.strip()

        except Exception as e:
            self.logger.warning(f"解析失败 {fna_file.name}: {e}")
            return self.species_name, fna_file.stem

    def parse_fna_for_species_name(self, fna_file: Path) -> str:
        """从.fna文件的第一行header解析菌株名（保持向后兼容）"""
        species, strain = self.parse_fna_for_species_and_strain(fna_file)
        return strain if strain else fna_file.stem

    def run_command(self, cmd: str, description: str) -> Optional[str]:
        """运行本地命令（不使用Docker）"""
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"Step: {description}")
        self.logger.debug(f"Command: {cmd}")
        self.logger.info(f"{'=' * 70}\n")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"❌ ERROR: {description} failed!")
            self.logger.error(f"STDERR: {result.stderr}")
            return None

        self.logger.info(f"✅ {description} completed")
        return result.stdout

    def step0_prepare_genomes(self) -> List[Path]:
        """Step 0: 准备和标准化基因组文件"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 0: Preparing and standardizing genome files")
        self.logger.info("=" * 70)

        genome_files = list(self.genome_dir.glob("*.fna")) + \
                       list(self.genome_dir.glob("*.fasta")) + \
                       list(self.genome_dir.glob("*.fa"))

        if not genome_files:
            self.logger.error(f"❌ 未找到基因组文件在 {self.genome_dir}")
            sys.exit(1)

        self.logger.info(f"找到 {len(genome_files)} 个基因组文件")

        std_genome_dir = self.output_dir / "standardized_genomes"
        std_genome_dir.mkdir(exist_ok=True)

        gcf_to_strain = {}
        genome_info = []

        for genome_file in genome_files:
            gcf_id = genome_file.stem.replace('_genomic', '')
            species_name, strain_name = self.parse_fna_for_species_and_strain(genome_file)

            if self.metadata_df is not None:
                match = self.metadata_df[self.metadata_df['accession'] == gcf_id]
                if not match.empty:
                    if 'strain' in match.columns:
                        metadata_strain = str(match['strain'].iloc[0])
                        if pd.notna(metadata_strain) and metadata_strain != 'N/A':
                            strain_name = metadata_strain
                    if 'organism_name' in match.columns:
                        org_name = str(match['organism_name'].iloc[0])
                        if pd.notna(org_name) and org_name != 'N/A':
                            words = org_name.split()
                            if len(words) >= 2:
                                species_name = f"{words[0]} {words[1]}"

            clean_strain = self._clean_strain_name(strain_name)

            original_strain = clean_strain
            counter = 1
            while clean_strain in gcf_to_strain.values():
                clean_strain = f"{original_strain}_{counter}"
                counter += 1

            gcf_to_strain[gcf_id] = clean_strain
            full_label = f"{species_name} {strain_name}"
            self.strain_to_full_label[clean_strain] = full_label

            records = list(SeqIO.parse(genome_file, "fasta"))
            total_length = sum(len(rec.seq) for rec in records)
            n_contigs = len(records)

            genome_info.append({
                'accession': gcf_id,
                'species_name': species_name,
                'strain_name': strain_name,
                'clean_strain_name': clean_strain,
                'full_label': full_label,
                'n_contigs': n_contigs,
                'total_length_bp': total_length,
                'size_mbp': round(total_length / 1e6, 2),
                'original_file': genome_file.name
            })

            std_file = std_genome_dir / f"{clean_strain}.fasta"
            shutil.copy2(genome_file, std_file)

            self.logger.info(f"  {gcf_id} → {clean_strain}")
            self.logger.info(f"      完整标签: {full_label} ({n_contigs} contigs, {total_length / 1e6:.2f} Mbp)")

        mapping_df = pd.DataFrame(genome_info)
        mapping_file = self.output_dir / "accession_to_strain_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)

        label_mapping_file = self.output_dir / "strain_label_mapping.csv"
        label_df = pd.DataFrame([
            {'strain': k, 'full_label': v}
            for k, v in self.strain_to_full_label.items()
        ])
        label_df.to_csv(label_mapping_file, index=False)

        self.logger.info(f"\n✅ 映射文件已保存: {mapping_file}")
        self.logger.info(f"✅ 标签映射文件: {label_mapping_file}")
        self.logger.info(f"✅ 标准化基因组目录: {std_genome_dir}")

        self.genome_dir = std_genome_dir

        return list(std_genome_dir.glob("*.fasta"))

    def _clean_strain_name(self, name: str) -> str:
        """清理菌株名称"""
        name = re.sub(r'[^\w\s\-\.]', '_', name)
        name = name.replace(' ', '_')
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        if len(name) > 50:
            name = name[:50]
        return name if name else "Unknown"

    def step1_calculate_ani_fastani(self, genomes: List[Path]) -> Path:
        """Step 1: 使用FastANI计算ANI"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 1: Calculating ANI with FastANI (Docker)")
        self.logger.info("=" * 70)

        genome_list = self.ani_results_dir / "genome_list.txt"
        with open(genome_list, 'w') as f:
            for genome in genomes:
                f.write(f"{genome.resolve()}\n")

        ani_output = self.ani_results_dir / "fastani_results.txt"
        cmd = f"fastANI --ql {genome_list} --rl {genome_list} -o {ani_output} -t {self.threads}"

        result = self._run_docker(
            self.docker_fastani, cmd,
            "FastANI calculation",
            mount_dirs=[self.ani_results_dir]
        )

        if result is None or not ani_output.exists():
            self.logger.error("❌ FastANI failed!")
            sys.exit(1)

        return ani_output

    def step2_parse_ani_results(self, ani_file: Path) -> pd.DataFrame:
        """Step 2: 解析ANI结果为矩阵"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 2: Parsing ANI results")
        self.logger.info("=" * 70)

        ani_data = []
        with open(ani_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query = Path(parts[0]).stem
                    ref = Path(parts[1]).stem
                    ani = float(parts[2])
                    ani_data.append({'query': query, 'reference': ref, 'ANI': ani})

        ani_df = pd.DataFrame(ani_data)
        strains = sorted(set(ani_df['query'].unique()) | set(ani_df['reference'].unique()))
        ani_matrix = pd.DataFrame(99.9, index=strains, columns=strains)

        for _, row in ani_df.iterrows():
            ani_matrix.loc[row['query'], row['reference']] = row['ANI']
            ani_matrix.loc[row['reference'], row['query']] = row['ANI']

        ani_matrix.to_csv(self.ani_results_dir / 'ani_matrix.csv')

        self.logger.info(f"ANI矩阵: {ani_matrix.shape[0]} x {ani_matrix.shape[1]}")
        self.logger.info(f"ANI范围: {ani_matrix.values.min():.2f}% - {ani_matrix.values.max():.2f}%")

        return ani_matrix

    def step3_filter_redundant_genomes(self, ani_matrix: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Step 3: ANI筛选去冗余"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"STEP 3: Filtering redundant genomes (ANI >= {self.ani_threshold}%)")
        self.logger.info("=" * 70)

        strains = list(ani_matrix.index)
        n_total = len(strains)

        redundant_pairs = []
        for i in range(len(strains)):
            for j in range(i + 1, len(strains)):
                ani = ani_matrix.iloc[i, j]
                if ani >= self.ani_threshold:
                    redundant_pairs.append((strains[i], strains[j], ani))

        self.logger.info(f"发现 {len(redundant_pairs)} 对冗余基因组 (ANI >= {self.ani_threshold}%)")

        if redundant_pairs:
            redundant_df = pd.DataFrame(redundant_pairs, columns=['Strain1', 'Strain2', 'ANI'])
            redundant_df.to_csv(self.ani_results_dir / 'redundant_pairs.csv', index=False)

        clusters = self._cluster_similar_genomes(strains, redundant_pairs)

        selected_strains = []
        removed_strains = []

        for cluster in clusters:
            rep = self._select_representative(cluster)
            selected_strains.append(rep)
            for strain in cluster:
                if strain != rep:
                    removed_strains.append(strain)

        self.logger.info(f"\n📊 筛选结果:")
        self.logger.info(f"  原始基因组数: {n_total}")
        self.logger.info(f"  保留基因组数: {len(selected_strains)}")
        self.logger.info(f"  移除基因组数: {len(removed_strains)}")

        pd.DataFrame({'strain': selected_strains}).to_csv(
            self.output_dir / 'selected_strains.csv', index=False)
        pd.DataFrame({'strain': removed_strains}).to_csv(
            self.output_dir / 'removed_strains.csv', index=False)

        return selected_strains, removed_strains

    def _cluster_similar_genomes(self, strains: List[str],
                                 redundant_pairs: List[Tuple]) -> List[List[str]]:
        """并查集聚类"""
        parent = {s: s for s in strains}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for s1, s2, _ in redundant_pairs:
            union(s1, s2)

        clusters = {}
        for s in strains:
            root = find(s)
            clusters.setdefault(root, []).append(s)

        return list(clusters.values())

    def _select_representative(self, cluster: List[str]) -> str:
        """选择聚类代表"""
        if len(cluster) == 1:
            return cluster[0]

        refs = ['SCgG2', 'ATCC_13032', 'ATCC_14067', 'ATCC_21799']
        for ref in refs:
            for strain in cluster:
                if ref in strain:
                    return strain

        return min(cluster, key=len)

    def step4_copy_selected_genomes(self, selected_strains: List[str]):
        """Step 4: 复制筛选后的基因组"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 4: Copying selected genomes")
        self.logger.info("=" * 70)

        copied = 0
        for strain in selected_strains:
            src = self.genome_dir / f"{strain}.fasta"
            dst = self.filtered_genomes_dir / f"{strain}.fasta"

            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                self.logger.warning(f"  ⚠️  文件不存在: {strain}")

        self.logger.info(f"✅ 复制了 {copied}/{len(selected_strains)} 个基因组")

    def step5_select_reference_genome(self, selected_strains: List[str]) -> Path:
        """Step 5: 选择参考基因组"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 5: Selecting reference genome")
        self.logger.info("=" * 70)

        preferred = ['SCgG2', 'ATCC_13032', 'ATCC_14067']

        ref_strain = None
        for pref in preferred:
            for strain in selected_strains:
                if pref in strain:
                    ref_strain = strain
                    break
            if ref_strain:
                break

        if not ref_strain:
            ref_strain = selected_strains[0]

        ref_path = self.filtered_genomes_dir / f"{ref_strain}.fasta"

        if not ref_path.exists():
            self.logger.error(f"❌ 参考基因组不存在: {ref_path}")
            sys.exit(1)

        self.logger.info(f"✅ 选择参考基因组: {ref_strain}")

        return ref_path

    def step6_snippy_calling(self, ref_genome: Path):
        """Step 6: Snippy SNP calling"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 6: SNP calling with Snippy (Docker)")
        self.logger.info("=" * 70)

        snippy_dir = self.output_dir / "snippy_results"
        snippy_dir.mkdir(exist_ok=True)

        genomes = list(self.filtered_genomes_dir.glob("*.fasta"))
        ref_stem = ref_genome.stem

        self.logger.info(f"对 {len(genomes) - 1} 个菌株运行Snippy")

        for genome in genomes:
            strain = genome.stem

            if strain == ref_stem:
                self.logger.info(f"  ⊘ 跳过参考: {strain}")
                continue

            outdir = snippy_dir / strain

            if outdir.exists():
                shutil.rmtree(outdir)

            self.logger.info(f"  → 处理: {strain}")

            cmd = f"snippy --force --outdir {outdir} --ref {ref_genome} --ctgs {genome} --cpus {self.threads}"
            result = self._run_docker(
                self.docker_snippy, cmd,
                f"Snippy-{strain}",
                mount_dirs=[snippy_dir, self.filtered_genomes_dir]
            )

            if result is None:
                self.logger.warning(f"  ⚠️  Snippy处理失败 {strain}")

    def step7_core_snps(self, ref_genome: Path):
        """Step 7: 提取Core SNPs"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 7: Extracting core SNPs (Docker)")
        self.logger.info("=" * 70)

        snippy_results_dir = self.output_dir / "snippy_results"

        snippy_dirs = [d for d in snippy_results_dir.glob("*")
                       if d.is_dir() and (d / "snps.vcf").exists()]

        if not snippy_dirs:
            self.logger.error("❌ 未找到有效的Snippy结果!")
            sys.exit(1)

        self.logger.info(f"找到 {len(snippy_dirs)} 个有效的Snippy结果目录")

        snippy_dirs_str = " ".join(str(d) for d in snippy_dirs)

        cmd = f"cd {self.output_dir} && snippy-core --ref {ref_genome} --prefix core {snippy_dirs_str}"
        result = self._run_docker(
            self.docker_snippy, cmd,
            "Snippy-core",
            mount_dirs=[snippy_results_dir]
        )

        if result is None:
            self.logger.error("❌ snippy-core 失败!")
            return

        core_aln = self.output_dir / "core.aln"
        core_snps_aln = self.output_dir / "core.snps.aln"

        if not core_aln.exists():
            self.logger.error(f"❌ core.aln 不存在")
            return

        cmd = f"snp-sites -c {core_aln} > {core_snps_aln}"
        self._run_docker(self.docker_snippy, cmd, "Extract SNP sites", mount_dirs=[])

        if core_snps_aln.exists():
            cmd = f"snp-sites -C {core_aln}"
            result = self._run_docker(self.docker_snippy, cmd, "SNP statistics", mount_dirs=[])
            if result:
                self.logger.info(f"\n{result}")

    def step8_build_tree(self, method: str = 'iqtree') -> Optional[Path]:
        """Step 8: 构建系统发育树"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"STEP 8: Building phylogenetic tree ({method}) (Docker)")
        self.logger.info("=" * 70)

        snp_aln = self.output_dir / "core.snps.aln"

        if not snp_aln.exists():
            self.logger.error(f"❌ SNP比对文件不存在: {snp_aln}")
            return None

        if method == 'iqtree':
            prefix = self.output_dir / "core_snps"
            cmd = f"iqtree -s {snp_aln} -m MFP -bb 1000 -nt {self.threads} -pre {prefix} -redo"

            self._run_docker(self.docker_iqtree, cmd, "IQ-TREE", mount_dirs=[])

            tree_file = self.output_dir / "core_snps.treefile"
        else:
            self.logger.error(f"未知的建树方法: {method}")
            return None

        if tree_file.exists():
            self.logger.info(f"✅ 系统发育树已生成: {tree_file}")
            self._create_labeled_tree(tree_file)
            return tree_file
        else:
            self.logger.error(f"❌ 树文件创建失败")
            return None

    def _create_labeled_tree(self, tree_file: Path):
        """创建带有完整物种名标签的树文件"""
        self.logger.info("\n创建带完整标签的树文件...")

        with open(tree_file, 'r') as f:
            tree_content = f.read()

        # 调试：打印树文件中的前几个标签和映射
        self.logger.info(f"  映射表中有 {len(self.strain_to_full_label)} 个条目")
        if self.strain_to_full_label:
            sample_items = list(self.strain_to_full_label.items())[:3]
            for strain, label in sample_items:
                self.logger.info(f"    映射示例: '{strain}' -> '{label}'")

        # 提取树文件中的标签（在newick格式中，标签在括号和冒号之间）
        tree_labels = re.findall(r'[(),]([A-Za-z0-9_.\-]+):', tree_content)
        if tree_labels:
            self.logger.info(f"  树文件中找到 {len(tree_labels)} 个标签")
            self.logger.info(f"    树标签示例: {tree_labels[:3]}")

        labeled_tree = tree_content
        replaced_count = 0

        # 按名称长度降序排列，避免短名称先被替换导致长名称无法匹配
        sorted_strains = sorted(self.strain_to_full_label.items(),
                                key=lambda x: len(x[0]), reverse=True)

        for strain, full_label in sorted_strains:
            # 将空格替换为下划线（newick格式不支持空格）
            safe_label = full_label.replace(' ', '_')
            # 在newick格式中，标签后面通常是 : ) , 或换行
            pattern = rf'(?<![A-Za-z0-9_]){re.escape(strain)}(?![A-Za-z0-9_])'

            # 检查是否能匹配到
            if re.search(pattern, labeled_tree):
                labeled_tree = re.sub(pattern, safe_label, labeled_tree)
                replaced_count += 1

        self.logger.info(f"  成功替换了 {replaced_count}/{len(self.strain_to_full_label)} 个标签")

        labeled_tree_file = self.output_dir / "core_snps.labeled.treefile"
        with open(labeled_tree_file, 'w') as f:
            f.write(labeled_tree)

        self.logger.info(f"✅ 带标签的树文件: {labeled_tree_file}")

    def run_full_pipeline(self):
        """运行完整流程"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🧬 Core SNPs Phylogenetic Tree Pipeline (Docker Version V2)")
        self.logger.info("=" * 70)
        self.logger.info(f"输入目录: {self.genome_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"ANI阈值: {self.ani_threshold}%")
        self.logger.info(f"默认物种名: {self.species_name}")

        try:
            genomes = self.step0_prepare_genomes()
            ani_file = self.step1_calculate_ani_fastani(genomes)
            ani_matrix = self.step2_parse_ani_results(ani_file)
            selected, removed = self.step3_filter_redundant_genomes(ani_matrix)
            self.step4_copy_selected_genomes(selected)
            ref_genome = self.step5_select_reference_genome(selected)
            self.step6_snippy_calling(ref_genome)
            self.step7_core_snps(ref_genome)
            tree_file = self.step8_build_tree(method='iqtree')

            self._print_final_summary(len(genomes), len(selected), len(removed))

        except KeyboardInterrupt:
            self.logger.warning("\n\n⚠️  用户中断流程")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"\n\n❌ 流程失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _print_final_summary(self, total: int, selected: int, removed: int):
        """打印最终总结"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🎉 流程完成!")
        self.logger.info("=" * 70)
        self.logger.info(f"原始基因组: {total}")
        self.logger.info(f"筛选后: {selected} (移除{removed}个)")
        self.logger.info(f"\n主要输出文件:")
        self.logger.info(f"  1. ANI矩阵: {self.ani_results_dir}/ani_matrix.csv")
        self.logger.info(f"  2. 筛选结果: {self.output_dir}/selected_strains.csv")
        self.logger.info(f"  3. 标签映射: {self.output_dir}/strain_label_mapping.csv")
        self.logger.info(f"  4. 系统发育树: {self.output_dir}/core_snps.treefile")
        self.logger.info(f"  5. 带标签的树: {self.output_dir}/core_snps.labeled.treefile")
        self.logger.info("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Complete Core SNPs phylogenetic tree pipeline (Docker Version V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python run_complete_pipeline_docker_v2.py -i genomes -o results

  # 指定物种名
  python run_complete_pipeline_docker_v2.py -i genomes -o results \\
      --species "Corynebacterium glutamicum"

  # 指定线程数和ANI阈值
  python run_complete_pipeline_docker_v2.py -i genomes -o results -t 16 --ani-threshold 99.9

环境准备:
  # 拉取Docker镜像
  docker pull staphb/fastani:latest
  docker pull staphb/snippy:latest
  docker pull staphb/iqtree:latest
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='输入目录（包含.fna基因组文件）')
    parser.add_argument('-o', '--output', default='snp_tree_results',
                        help='输出目录 (default: snp_tree_results)')
    parser.add_argument('-t', '--threads', type=int, default=8,
                        help='线程数 (default: 8)')
    parser.add_argument('--ani-threshold', type=float, default=99.9,
                        help='ANI阈值 (default: 99.9)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='元数据CSV文件路径（可选）')
    parser.add_argument('--species', type=str,
                        default='Corynebacterium glutamicum',
                        help='默认物种名 (default: Corynebacterium glutamicum)')

    # Docker镜像配置
    parser.add_argument('--docker-fastani', type=str,
                        default='staphb/fastani:latest',
                        help='FastANI Docker镜像')
    parser.add_argument('--docker-snippy', type=str,
                        default='staphb/snippy:latest',
                        help='Snippy Docker镜像')
    parser.add_argument('--docker-iqtree', type=str,
                        default='staphb/iqtree:latest',
                        help='IQ-TREE Docker镜像')

    args = parser.parse_args()

    pipeline = SNPTreePipelineDocker(
        genome_dir=args.input,
        output_dir=args.output,
        metadata_file=args.metadata,
        threads=args.threads,
        ani_threshold=args.ani_threshold,
        species_name=args.species,
        docker_fastani=args.docker_fastani,
        docker_snippy=args.docker_snippy,
        docker_iqtree=args.docker_iqtree
    )

    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()