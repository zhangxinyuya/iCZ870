"""
Energy Generating Cycle (EGC) Detector with Auto-Fix
=====================================================
检测通用模型引入后可能产生的能量生成循环，并自动修复

新增功能：
- 检测到EGC后，自动找到负通量的可逆反应
- 将这些反应的可逆性改为不可逆（禁止反向反应）
- 迭代修复直到没有EGC

Author: Claude
Version: 3.0 (with Auto-Fix support)
"""

import cobra
from cobra import Model, Reaction, Metabolite
from cobra.flux_analysis import loopless_solution
from optlang.symbolics import Zero
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import os
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')


class EGCDetectorAutoFix:
    """
    能量生成循环（EGC）检测器 - 带自动修复功能

    工作流程：
    1. 检测EGC
    2. 找到EGC中通量为负的可逆反应
    3. 将这些反应改为不可逆（lower_bound = 0）
    4. 重复检测直到没有EGC
    """

    def __init__(
        self,
        user_model: Model,
        universal_model: Model,
        atp_reaction: str = 'rxn00062_c',
        max_universal_reactions: int = 5,
        flux_threshold: float = 1e-6,
        big_m: float = 1000.0,
        max_iterations: int = 1000,
        glucose_exchange: str = 'EX_cpd00027_e',
        use_loopless: bool = True
    ):
        """
        Parameters:
        -----------
        user_model : cobra.Model
            用户模型（基础模型）
        universal_model : cobra.Model
            通用模型（候选反应来源）
        atp_reaction : str
            ATP合成反应ID
        max_universal_reactions : int
            最多允许参与的通用模型反应数量
        flux_threshold : float
            通量阈值
        big_m : float
            Big-M约束值
        max_iterations : int
            最大迭代次数
        glucose_exchange : str
            葡萄糖交换反应ID
        use_loopless : bool
            是否使用loopless FBA
        """
        self.user_model = user_model.copy()
        self.universal_model = universal_model.copy()
        self.atp_reaction = atp_reaction
        self.max_universal_reactions = max_universal_reactions
        self.flux_threshold = flux_threshold
        self.big_m = big_m
        self.max_iterations = max_iterations
        self.glucose_exchange = glucose_exchange
        self.use_loopless = use_loopless

        # 结果存储
        self.egc_list = []
        self.candidate_reactions = []
        self.binary_variables = {}
        self.combined_model = None
        self.false_positive_count = 0

        # 修复记录
        self.fixed_reactions = []  # 记录所有被修复的反应

    def _log(self, msg: str):
        print(msg)

    def _get_user_reaction_ids(self) -> Set[str]:
        """获取用户模型的反应ID集合"""
        ids = set()
        for rxn in self.user_model.reactions:
            ids.add(rxn.id)
            base_id = rxn.id.replace('_forward', '').replace('_backward', '')
            ids.add(base_id)
        return ids

    def _get_user_metabolite_ids(self) -> Set[str]:
        """获取用户模型的代谢物ID集合"""
        return {met.id for met in self.user_model.metabolites}

    def _find_candidate_reactions(self) -> List[Reaction]:
        """从通用模型中筛选候选反应"""
        user_rxn_ids = self._get_user_reaction_ids()
        user_met_ids = self._get_user_metabolite_ids()

        candidates = []

        for rxn in self.universal_model.reactions:
            base_id = rxn.id.replace('_forward', '').replace('_backward', '')

            if rxn.id in user_rxn_ids or base_id in user_rxn_ids:
                continue

            if rxn.id.startswith(('EX_', 'DM_', 'SK_')):
                continue

            rxn_met_ids = {met.id for met in rxn.metabolites}
            if rxn_met_ids & user_met_ids:
                candidates.append(rxn)

        self._log(f"  筛选出 {len(candidates)} 个候选反应")
        self.candidate_reactions = candidates
        return candidates

    def _build_milp_model(self) -> Model:
        """构建用于EGC检测的MILP模型"""
        self._log("  构建MILP模型...")

        combined = self.user_model.copy()

        candidate_ids = []
        for rxn in self.candidate_reactions:
            if rxn.id not in [r.id for r in combined.reactions]:
                new_rxn = rxn.copy()
                for met in rxn.metabolites:
                    if met.id not in [m.id for m in combined.metabolites]:
                        combined.add_metabolites([met.copy()])
                combined.add_reactions([new_rxn])
                candidate_ids.append(rxn.id)

        self._log(f"    添加了 {len(candidate_ids)} 个候选反应")

        # 限制通量范围
        for rxn in combined.reactions:
            if rxn.reversibility:
                rxn.lower_bound = max(rxn.lower_bound, -1)
                rxn.upper_bound = min(rxn.upper_bound, 1)
            else:
                rxn.lower_bound = max(rxn.lower_bound, 0)
                rxn.upper_bound = min(rxn.upper_bound, 1)

        # 关闭葡萄糖交换
        if self.glucose_exchange in [r.id for r in combined.reactions]:
            glc_rxn = combined.reactions.get_by_id(self.glucose_exchange)
            glc_rxn.lower_bound = 0
            glc_rxn.upper_bound = 0
            self._log(f"    已关闭碳源: {self.glucose_exchange}")

        # ATP反应设置
        if self.atp_reaction in [r.id for r in combined.reactions]:
            atp_rxn = combined.reactions.get_by_id(self.atp_reaction)
            atp_rxn.lower_bound = -1000
            atp_rxn.upper_bound = 1000
        else:
            if self.atp_reaction in [r.id for r in self.universal_model.reactions]:
                atp_rxn = self.universal_model.reactions.get_by_id(self.atp_reaction).copy()
                for met in atp_rxn.metabolites:
                    if met.id not in [m.id for m in combined.metabolites]:
                        combined.add_metabolites([met.copy()])
                combined.add_reactions([atp_rxn])
                atp_rxn = combined.reactions.get_by_id(self.atp_reaction)
                atp_rxn.lower_bound = -1000
                atp_rxn.upper_bound = 1000
            else:
                raise ValueError(f"ATP反应 {self.atp_reaction} 不存在！")

        combined.objective = self.atp_reaction

        # 创建二进制变量
        solver = combined.solver
        self.binary_variables = {}

        for rxn_id in candidate_ids:
            rxn = combined.reactions.get_by_id(rxn_id)

            y_var = solver.interface.Variable(
                name=f"y_{rxn_id}",
                type="binary"
            )
            solver.add(y_var)
            self.binary_variables[rxn_id] = y_var

            v_fwd = rxn.forward_variable
            v_rev = rxn.reverse_variable

            solver.add(solver.interface.Constraint(
                v_fwd - self.big_m * y_var,
                ub=0,
                name=f"bigM_fwd_{rxn_id}"
            ))

            solver.add(solver.interface.Constraint(
                v_rev - self.big_m * y_var,
                ub=0,
                name=f"bigM_rev_{rxn_id}"
            ))

        y_sum = Zero
        for y_var in self.binary_variables.values():
            y_sum = y_sum + y_var

        solver.add(solver.interface.Constraint(
            y_sum,
            ub=self.max_universal_reactions,
            name="max_reactions_constraint"
        ))

        self.combined_model = combined
        self._log(f"    模型构建完成: {len(combined.reactions)} 反应")

        return combined

    def _solve_milp(self) -> Tuple[float, List[str], Dict[str, float]]:
        """求解MILP"""
        model = self.combined_model

        try:
            self._log("      [DEBUG] 开始MILP求解...")
            solution = model.optimize()
            self._log("      [DEBUG] MILP求解完成")
        except Exception as e:
            self._log(f"    MILP求解失败: {e}")
            return 0.0, [], {}

        if solution.status != 'optimal':
            return 0.0, [], {}

        milp_atp_flux = abs(solution.objective_value)

        # ✅ 修复1：使用二进制变量判断选中的反应
        selected_rxns = []
        for rxn_id, y_var in self.binary_variables.items():
            try:
                if y_var.primal is not None and y_var.primal > 0.5:
                    selected_rxns.append(rxn_id)
            except:
                flux = solution.fluxes.get(rxn_id, 0)
                if abs(flux) > self.flux_threshold:
                    selected_rxns.append(rxn_id)

        if milp_atp_flux < self.flux_threshold or not selected_rxns:
            return 0.0, [], {}

        if self.use_loopless:
            self._log("      [DEBUG] 开始Loopless验证...")
            atp_flux, active_fluxes = self._verify_with_loopless(selected_rxns)
            self._log("      [DEBUG] Loopless验证完成")
        else:
            atp_flux = milp_atp_flux

            # ✅ 修复2：只收集选中反应的通量，而不是所有反应
            active_fluxes = {}
            for rxn_id in selected_rxns:
                flux = solution.fluxes.get(rxn_id, 0)
                if abs(flux) > self.flux_threshold:
                    active_fluxes[rxn_id] = flux

            # 可选：也包含ATP反应的通量
            if self.atp_reaction in [r.id for r in model.reactions]:
                atp_flux_val = solution.fluxes.get(self.atp_reaction, 0)
                if abs(atp_flux_val) > self.flux_threshold:
                    active_fluxes[self.atp_reaction] = atp_flux_val

        return atp_flux, selected_rxns, active_fluxes

    def _verify_with_loopless(self, selected_rxns: List[str]) -> Tuple[float, Dict[str, float]]:
        """使用loopless FBA验证"""
        test_model = self.user_model.copy()

        for rxn_id in selected_rxns:
            if rxn_id not in [r.id for r in test_model.reactions]:
                try:
                    rxn = self.universal_model.reactions.get_by_id(rxn_id)
                    new_rxn = rxn.copy()
                    for met in rxn.metabolites:
                        if met.id not in [m.id for m in test_model.metabolites]:
                            test_model.add_metabolites([met.copy()])
                    test_model.add_reactions([new_rxn])
                except KeyError:
                    continue

        for rxn in test_model.reactions:
            if rxn.reversibility:
                rxn.lower_bound = max(rxn.lower_bound, -1)
                rxn.upper_bound = min(rxn.upper_bound, 1)
            else:
                rxn.lower_bound = max(rxn.lower_bound, 0)
                rxn.upper_bound = min(rxn.upper_bound, 1)

        if self.glucose_exchange in [r.id for r in test_model.reactions]:
            glc_rxn = test_model.reactions.get_by_id(self.glucose_exchange)
            glc_rxn.lower_bound = 0
            glc_rxn.upper_bound = 0

        if self.atp_reaction in [r.id for r in test_model.reactions]:
            atp_rxn = test_model.reactions.get_by_id(self.atp_reaction)
            atp_rxn.lower_bound = -1000
            atp_rxn.upper_bound = 1000
        else:
            try:
                atp_rxn = self.universal_model.reactions.get_by_id(self.atp_reaction).copy()
                for met in atp_rxn.metabolites:
                    if met.id not in [m.id for m in test_model.metabolites]:
                        test_model.add_metabolites([met.copy()])
                test_model.add_reactions([atp_rxn])
                atp_rxn = test_model.reactions.get_by_id(self.atp_reaction)
                atp_rxn.lower_bound = -1000
                atp_rxn.upper_bound = 1000
            except:
                return 0.0, {}

        test_model.objective = self.atp_reaction

        try:
            solution = loopless_solution(test_model)
        except Exception:
            try:
                solution = test_model.optimize()
            except:
                return 0.0, {}

        if solution.status != 'optimal':
            return 0.0, {}

        atp_flux = abs(solution.objective_value)

        active_fluxes = {}
        for rxn in test_model.reactions:
            flux = solution.fluxes.get(rxn.id, 0)
            if abs(flux) > self.flux_threshold:
                active_fluxes[rxn.id] = flux

        return atp_flux, active_fluxes

    def _add_integer_cut(self, selected_rxns: List[str], cut_id: int):
        """添加整数割"""
        solver = self.combined_model.solver

        cut_expr = Zero
        for rxn_id in selected_rxns:
            if rxn_id in self.binary_variables:
                cut_expr = cut_expr + self.binary_variables[rxn_id]

        solver.add(solver.interface.Constraint(
            cut_expr,
            ub=len(selected_rxns) - 1,
            name=f"integer_cut_{cut_id}"
        ))

    def detect_and_fix_all(self) -> Tuple[List[Dict], List[Dict]]:
        """
        迭代检测并修复所有EGC

        优化策略：
        1. 一次构建MILP模型
        2. 检测到EGC后，直接在MILP模型中修改负通量反应的下限为0
        3. 继续检测，无需重建模型
        4. 最后同步修改到通用模型

        Returns:
        --------
        Tuple[List[Dict], List[Dict]]:
            (所有发现的EGC列表, 所有修复的反应列表)
        """
        print("=" * 70)
        print("能量生成循环（EGC）检测器 - 自动修复模式")
        print("=" * 70)
        print(f"ATP检测反应: {self.atp_reaction}")
        print(f"关闭的碳源: {self.glucose_exchange}")
        print(f"最大通用反应数: {self.max_universal_reactions}")
        print(f"检测方法: {'Loopless FBA' if self.use_loopless else '普通FBA'}")
        print()

        # Step 1: 筛选候选反应
        print(">>> Step 1: 筛选候选反应")
        self._find_candidate_reactions()

        if not self.candidate_reactions:
            print("✓ 没有候选反应，无需检测")
            return [], []
        print()

        # Step 2: 一次性构建MILP模型
        print(">>> Step 2: 构建MILP模型（仅一次）")
        self._build_milp_model()
        print()

        # Step 3: 迭代检测EGC，直接在模型中修改边界
        print(">>> Step 3: 迭代检测并修复EGC")
        print("-" * 50)

        all_egcs = []
        fixed_rxn_ids = set()  # 已修复的反应ID
        egc_count = 0
        no_fix_count = 0  # 连续找不到可修复反应的次数

        for iteration in range(self.max_iterations):
            # 求解MILP
            atp_flux, selected_rxns, active_fluxes = self._solve_milp()

            # 检查是否有候选反应被选中
            if not selected_rxns:
                print(f"  迭代 {iteration + 1}: 无候选反应被选中，检测完成")
                break

            # 检查是否有ATP产生
            if atp_flux < self.flux_threshold:
                if self.use_loopless:
                    self.false_positive_count += 1
                    print(f"  迭代 {iteration + 1}: ○ 假阳性 (Loopless验证ATP=0)")
                    self._add_integer_cut(selected_rxns, iteration)
                    continue
                else:
                    print(f"  迭代 {iteration + 1}: 无ATP产生，检测完成")
                    break

            # 发现EGC
            egc_count += 1
            print(f"  迭代 {iteration + 1}: ✗ 发现 EGC #{egc_count}")
            print(f"    ATP通量: {atp_flux:.6f}")
            print(f"    涉及通用反应: {', '.join(selected_rxns)}")

            # 收集EGC信息
            rxn_details = []
            for rxn_id in selected_rxns:
                try:
                    rxn = self.universal_model.reactions.get_by_id(rxn_id)
                    flux = active_fluxes.get(rxn_id, 0)
                    rxn_details.append({
                        'id': rxn_id,
                        'equation': rxn.reaction,
                        'flux': flux,
                        'name': rxn.name if rxn.name else ''
                    })
                except:
                    rxn_details.append({
                        'id': rxn_id,
                        'equation': 'N/A',
                        'flux': active_fluxes.get(rxn_id, 0),
                        'name': ''
                    })

            egc_info = {
                'egc_id': egc_count,
                'atp_flux': atp_flux,
                'universal_reactions': selected_rxns.copy(),
                'universal_reaction_details': rxn_details,
                'all_active_fluxes': active_fluxes.copy(),
                'num_universal': len(selected_rxns)
            }
            all_egcs.append(egc_info)

            # 找到需要修复的反应并直接在MILP模型中修改
            new_fixes = self._find_and_fix_in_milp(active_fluxes, fixed_rxn_ids)

            if new_fixes:
                no_fix_count = 0
                for fix_info in new_fixes:
                    print(f"    → 修复: {fix_info['rxn_id']} (flux={fix_info['flux']:.6f})")
                    fixed_rxn_ids.add(fix_info['rxn_id'])
            else:
                # 没有可修复的反应，用整数割排除这个组合
                no_fix_count += 1
                print(f"    → 无可修复的通用模型反应，添加整数割")
                self._add_integer_cut(selected_rxns, egc_count)

                # 如果连续多次找不到可修复的反应，可能陷入死循环
                if no_fix_count >= 10:
                    print(f"  警告: 连续{no_fix_count}次无法修复，停止检测")
                    break

            print()

        print("-" * 50)
        print()

        # Step 4: 同步修改到通用模型
        print(">>> Step 4: 同步修改到通用模型")
        self._sync_fixes_to_universal_model()
        print(f"  已同步 {len(self.fixed_reactions)} 个反应的修改")
        print()

        # 最终总结
        print("=" * 70)
        print("修复总结")
        print("=" * 70)
        print(f"检测到的EGC总数: {len(all_egcs)}")
        print(f"修复的反应总数: {len(self.fixed_reactions)}")

        if self.fixed_reactions:
            print()
            print("修复的反应列表（通用模型）:")
            for i, fix in enumerate(self.fixed_reactions, 1):
                print(f"  {i}. {fix['rxn_id']}")
                print(f"     方程: {fix['equation']}")
                print(f"     边界: [{fix['original_lb']}, {fix['original_ub']}] → [0, {fix['original_ub']}]")

        print("=" * 70)

        self.egc_list = all_egcs
        return all_egcs, self.fixed_reactions

    def _find_and_fix_in_milp(
            self,
            active_fluxes: Dict[str, float],
            already_fixed: Set[str]
    ) -> List[Dict]:
        """
        找到通用模型中参与EGC的可逆反应，修改其边界

        策略：
        1. 优先找所有负通量的可逆反应，设置 lower_bound = 0
        2. 如果没有负通量反应，找所有正通量的可逆反应，设置 upper_bound = 0
        3. 一次性修复所有符合条件的反应

        Parameters:
        -----------
        active_fluxes : Dict[str, float]
            活跃反应的通量
        already_fixed : Set[str]
            已经修复过的反应ID

        Returns:
        --------
        List[Dict]: 本次修复的反应列表
        """
        new_fixes = []

        # 分类：负通量 vs 正通量的通用模型可逆反应
        negative_flux_candidates = []
        positive_flux_candidates = []

        for rxn_id, flux in active_fluxes.items():
            # 跳过ATP反应
            if rxn_id == self.atp_reaction:
                continue

            # 跳过已修复的
            if rxn_id in already_fixed:
                continue

            # 只处理通用模型中的反应
            if rxn_id not in [r.id for r in self.universal_model.reactions]:
                continue

            # 检查在MILP模型中是否存在
            if rxn_id not in [r.id for r in self.combined_model.reactions]:
                continue

            milp_rxn = self.combined_model.reactions.get_by_id(rxn_id)
            universal_rxn = self.universal_model.reactions.get_by_id(rxn_id)

            if flux < -self.flux_threshold and milp_rxn.lower_bound < 0:
                # 负通量且可以禁止反向
                negative_flux_candidates.append((rxn_id, flux, milp_rxn, universal_rxn))
            elif flux > self.flux_threshold and milp_rxn.upper_bound > 0:
                # 正通量且可以禁止正向
                positive_flux_candidates.append((rxn_id, flux, milp_rxn, universal_rxn))

        # 策略：优先修复所有负通量，没有则修复所有正通量
        if negative_flux_candidates:
            candidates_to_fix = negative_flux_candidates
            fix_type = 'negative'
        elif positive_flux_candidates:
            candidates_to_fix = positive_flux_candidates
            fix_type = 'positive'
        else:
            return []

        # 一次性修复所有候选反应
        for rxn_id, flux, milp_rxn, universal_rxn in candidates_to_fix:
            if fix_type == 'negative':
                fix_info = {
                    'rxn_id': rxn_id,
                    'flux': flux,
                    'fix_type': 'block_reverse',
                    'original_lb': milp_rxn.lower_bound,
                    'original_ub': milp_rxn.upper_bound,
                    'new_lb': 0,
                    'new_ub': milp_rxn.upper_bound,
                    'equation': universal_rxn.reaction,
                    'name': universal_rxn.name if universal_rxn.name else '',
                }
                milp_rxn.lower_bound = 0
            else:
                fix_info = {
                    'rxn_id': rxn_id,
                    'flux': flux,
                    'fix_type': 'block_forward',
                    'original_lb': milp_rxn.lower_bound,
                    'original_ub': milp_rxn.upper_bound,
                    'new_lb': milp_rxn.lower_bound,
                    'new_ub': 0,
                    'equation': universal_rxn.reaction,
                    'name': universal_rxn.name if universal_rxn.name else '',
                }
                milp_rxn.upper_bound = 0

            self.fixed_reactions.append(fix_info)
            new_fixes.append(fix_info)

        return new_fixes

    def _sync_fixes_to_universal_model(self):
        """将修复同步到通用模型"""
        for fix_info in self.fixed_reactions:
            rxn_id = fix_info['rxn_id']
            if rxn_id in [r.id for r in self.universal_model.reactions]:
                rxn = self.universal_model.reactions.get_by_id(rxn_id)
                if fix_info['fix_type'] == 'block_reverse':
                    rxn.lower_bound = 0
                else:  # block_forward
                    rxn.upper_bound = 0

    def get_fixed_universal_model(self) -> Model:
        """
        获取修复后的通用模型

        Returns:
        --------
        Model: 修复后的通用模型
        """
        return self.universal_model.copy()

    def save_results(self, output_path: str = './egc_fixed_reactions.csv'):
        """保存修复结果（仅保存修复的反应列表）"""
        if not self.fixed_reactions:
            print("没有修复的反应需要保存")
            return None

        fix_rows = []
        for fix in self.fixed_reactions:
            fix_rows.append({
                'Reaction_ID': fix['rxn_id'],
                'Original_LB': fix['original_lb'],
                'Original_UB': fix['original_ub'],
                'New_LB': 0,
                'New_UB': fix['original_ub'],
                'Equation': fix['equation'],
                'Name': fix['name'],
                'Flux_When_Detected': fix['flux']
            })

        fix_df = pd.DataFrame(fix_rows)
        fix_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"修复记录已保存到: {output_path}")

        return fix_df

    def save_fixed_universal_model(self, output_path: str):
        """
        保存修复后的通用模型

        Parameters:
        -----------
        output_path : str
            通用模型输出路径
        """
        if output_path.endswith('.xml') or output_path.endswith('.sbml'):
            cobra.io.write_sbml_model(self.universal_model, output_path)
        elif output_path.endswith('.json'):
            cobra.io.save_json_model(self.universal_model, output_path)
        print(f"修复后的通用模型已保存到: {output_path}")


def load_model(filepath: str) -> Model:
    """加载COBRA模型"""
    if filepath.endswith('.xml') or filepath.endswith('.sbml'):
        return cobra.io.read_sbml_model(filepath)
    elif filepath.endswith('.json'):
        return cobra.io.load_json_model(filepath)
    else:
        raise ValueError(f"不支持的格式: {filepath}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='能量生成循环（EGC）检测器 - 自动修复版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python egc_detector_autofix.py -u user_model.xml -r universal.json -o fixed_reactions.csv

  # 保存修复后的通用模型
  python egc_detector_autofix.py -u user.xml -r universal.json --save-universal fixed_universal.json

修复原理:
  1. 检测EGC
  2. 找到EGC中通量为负的可逆反应（仅限通用模型）
  3. 将这些反应的lower_bound设为0（禁止反向）
  4. 重复直到没有EGC
  
注意：只修改通用模型中的反应，不修改用户模型。
        """
    )

    parser.add_argument('-u', '--user-model', required=True, help='用户模型文件')
    parser.add_argument('-r', '--universal-model', required=True, help='通用模型文件')
    parser.add_argument('-a', '--atp-reaction', default='rxn00062_c', help='ATP合成反应ID')
    parser.add_argument('-g', '--glucose-exchange', default='EX_cpd00027_e', help='葡萄糖交换反应ID')
    parser.add_argument('-m', '--max-reactions', type=int, default=5, help='最大通用反应数')
    parser.add_argument('-t', '--threshold', type=float, default=1e-6, help='通量阈值')
    parser.add_argument('-i', '--max-iterations', type=int, default=10000, help='最大迭代次数（默认10000）')
    parser.add_argument('-o', '--output', default='./egc_fixed_reactions.csv', help='修复记录输出路径')
    parser.add_argument('--save-universal', help='保存修复后的通用模型路径')
    parser.add_argument('--no-loopless', action='store_true', help='不使用loopless FBA')

    args = parser.parse_args()

    print("加载模型...")
    user_model = load_model(args.user_model)
    universal_model = load_model(args.universal_model)
    print(f"用户模型: {len(user_model.reactions)} 反应")
    print(f"通用模型: {len(universal_model.reactions)} 反应")
    print()

    detector = EGCDetectorAutoFix(
        user_model=user_model,
        universal_model=universal_model,
        atp_reaction=args.atp_reaction,
        max_universal_reactions=args.max_reactions,
        flux_threshold=args.threshold,
        glucose_exchange=args.glucose_exchange,
        use_loopless=not args.no_loopless,
        max_iterations=args.max_iterations
    )

    # 执行检测和修复
    egcs, fixes = detector.detect_and_fix_all()

    # 保存修复记录
    detector.save_results(args.output)

    # 保存修复后的通用模型
    if args.save_universal:
        detector.save_fixed_universal_model(args.save_universal)


if __name__ == "__main__":
    main()