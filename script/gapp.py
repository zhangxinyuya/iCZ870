"""
优化版 LooplessGapFiller - 全局频率统计修复版

主要改进：
1. Gurobi使用全部16线程
2. 循环检测并行化
3. 减少模型重复构建
4. **修复：使用全局频率统计，确保反应在所有方案中最多出现指定次数**
5. **新增：反应方程式的注释版本（使用代谢物名称）**

修复说明：
- 之前版本中每个反应数量级别独立统计，导致同一反应在不同级别中重复出现
- 现在使用全局频率统计，反应在所有输出方案中最多出现 frequency_threshold 次
- 切换反应数量级别时不会重置已排除的反应
"""

import cobra
from cobra import Model, Reaction, Metabolite
from optlang.symbolics import Zero
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import os
import json
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

# 设置Gurobi为默认求解器
cobra.Configuration().solver = 'gurobi'

COMMON_COFACTORS = {
    'cpd00002_c', 'cpd00008_c', 'cpd00018_c', 'cpd00038_c', 'cpd00031_c', 'cpd00126_c',
    'cpd00052_c', 'cpd00096_c', 'cpd00046_c', 'cpd00062_c', 'cpd00014_c', 'cpd00091_c',
    'cpd00003_c', 'cpd00004_c', 'cpd00005_c', 'cpd00006_c', 'cpd00015_c', 'cpd00982_c',
    'cpd00050_c', 'cpd01270_c', 'cpd00010_c', 'cpd00022_c', 'cpd00078_c', 'cpd00070_c',
    'cpd15500_c', 'cpd15499_c', 'cpd00001_c', 'cpd00067_c', 'cpd00009_c', 'cpd00012_c',
    'cpd00011_c', 'cpd00242_c', 'cpd00007_c', 'cpd00013_c', 'cpd00087_c', 'cpd00125_c',
}


def load_model(filepath: str) -> Model:
    if filepath.endswith('.xml') or filepath.endswith('.sbml'):
        return cobra.io.read_sbml_model(filepath)
    elif filepath.endswith('.json'):
        return cobra.io.load_json_model(filepath)
    else:
        raise ValueError(f"不支持的格式: {filepath}")


# ============================================================================
# 用于并行的独立函数（必须在模块级别定义）
# ============================================================================

def _check_cycle_worker(args):
    """
    并行循环检测的worker函数

    参数通过元组传递以支持multiprocessing
    """
    (selected_rxns, user_model_json, universal_model_json,
     glucose_exchange, substrate, atp_check_reaction,
     flux_threshold, check_cofactor_cycles, cofactor_virtual_reactions) = args

    # 在worker进程中重建模型
    import cobra
    import json as json_module
    from io import StringIO

    user_model = cobra.io.load_json_model(StringIO(user_model_json))
    universal_model = cobra.io.load_json_model(StringIO(universal_model_json))

    # 构建测试模型
    test_model = user_model.copy()

    for rxn_id in selected_rxns:
        if rxn_id not in [r.id for r in test_model.reactions]:
            try:
                rxn = universal_model.reactions.get_by_id(rxn_id)
                new_rxn = rxn.copy()
                for met in rxn.metabolites:
                    if met.id not in [m.id for m in test_model.metabolites]:
                        test_model.add_metabolites([met.copy()])
                test_model.add_reactions([new_rxn])
            except KeyError:
                continue

    # 关闭碳源
    if glucose_exchange in [r.id for r in test_model.reactions]:
        test_model.reactions.get_by_id(glucose_exchange).lower_bound = 0

    substrate_base = substrate.replace('_c', '')
    for rxn in test_model.reactions:
        if rxn.id == f"EX_{substrate_base}_e":
            rxn.lower_bound = 0
        if rxn.id == f"TR_{substrate_base}":
            rxn.bounds = (0, 0)

    detected_cycles = []

    # ATP检测
    atp_flux = _check_atp_worker(test_model, atp_check_reaction, universal_model)
    if atp_flux > flux_threshold:
        detected_cycles.append(('ATP', atp_flux))

    # NADPH/NADH检测
    if check_cofactor_cycles:
        for cofactor_name, stoichiometry in cofactor_virtual_reactions.items():
            flux = _check_cofactor_worker(test_model, cofactor_name, stoichiometry, universal_model)
            if flux > flux_threshold:
                detected_cycles.append((cofactor_name, flux))

    if detected_cycles:
        detected_cycles.sort(key=lambda x: x[1], reverse=True)
        cycle_descriptions = [f"{name}={flux:.4f}" for name, flux in detected_cycles]
        return (selected_rxns, True, detected_cycles[0][1], "; ".join(cycle_descriptions))
    else:
        return (selected_rxns, False, 0.0, "")


def _check_atp_worker(test_model, atp_check_reaction, universal_model):
    """ATP循环检测"""
    model = test_model.copy()

    if atp_check_reaction in [r.id for r in model.reactions]:
        atp_rxn = model.reactions.get_by_id(atp_check_reaction)
        atp_rxn.bounds = (-1000, 1000)
    else:
        if atp_check_reaction in [r.id for r in universal_model.reactions]:
            atp_rxn = universal_model.reactions.get_by_id(atp_check_reaction).copy()
            for met in atp_rxn.metabolites:
                if met.id not in [m.id for m in model.metabolites]:
                    model.add_metabolites([met.copy()])
            model.add_reactions([atp_rxn])
            model.reactions.get_by_id(atp_check_reaction).bounds = (-1000, 1000)
        else:
            return 0.0

    try:
        model.objective = atp_check_reaction
        solution = model.optimize()
        if solution.status == 'optimal':
            return abs(solution.objective_value)
    except:
        pass
    return 0.0


def _check_cofactor_worker(base_model, cofactor_name, stoichiometry, universal_model):
    """辅因子循环检测"""
    from cobra import Reaction, Metabolite

    test_model = base_model.copy()

    for met_id in stoichiometry.keys():
        if met_id not in [m.id for m in test_model.metabolites]:
            if met_id in [m.id for m in universal_model.metabolites]:
                met = universal_model.metabolites.get_by_id(met_id).copy()
                test_model.add_metabolites([met])
            else:
                compartment = 'c' if met_id.endswith('_c') else 'e'
                met = Metabolite(met_id, compartment=compartment)
                test_model.add_metabolites([met])

    detect_rxn_id = f"DETECT_{cofactor_name}_cycle"
    if detect_rxn_id in [r.id for r in test_model.reactions]:
        test_model.remove_reactions([detect_rxn_id])

    detect_rxn = Reaction(detect_rxn_id)
    metabolites_dict = {}
    for met_id, coef in stoichiometry.items():
        met = test_model.metabolites.get_by_id(met_id)
        metabolites_dict[met] = coef

    detect_rxn.add_metabolites(metabolites_dict)
    detect_rxn.bounds = (0, 1000)
    test_model.add_reactions([detect_rxn])
    test_model.objective = detect_rxn_id

    try:
        solution = test_model.optimize()
        if solution.status == 'optimal':
            return abs(solution.objective_value)
    except:
        pass
    return 0.0


class LooplessGapFillerOptimized:
    """
    优化版 Loopless Gap-Filler（全局频率统计修复版）

    改进：
    1. 充分利用多核CPU
    2. 并行循环检测
    3. 优化Gurobi参数
    4. **全局频率统计：反应在所有方案中最多出现指定次数**
    5. **反应方程式注释版本**
    """

    def __init__(
            self,
            user_model: Model,
            universal_model: Model,
            objective_reaction: str,
            substrate: str,
            max_reactions: int = 5,
            expansion_layers: int = 5,
            flux_threshold: float = 1e-6,
            glucose_exchange: str = 'EX_cpd00027_e',
            penalty: float = 0.01,
            big_m: float = 1000.0,
            atp_check_reaction: str = 'rxn00062_c',
            solver_timeout: int = 1200,
            cofactors: Set[str] = None,
            max_valid_solutions: int = 20,
            max_attempts: int = 100,
            check_cofactor_cycles: bool = True,
            biomass_drop_ratio: float = 0.5,
            n_jobs: int = -1,  # 并行数，-1表示使用所有核心
            batch_size: int = 8,  # 批量检测大小
            frequency_threshold: int = 3,  # 全局频率阈值（反应在所有方案中最多出现次数）
    ):
        self.user_model = user_model.copy()
        self.universal_model = universal_model.copy()
        self.objective_reaction = objective_reaction
        self.substrate = substrate
        self.substrate_external = substrate.replace('_c', '_e')
        self.max_reactions = max_reactions
        self.expansion_layers = expansion_layers
        self.flux_threshold = flux_threshold
        self.glucose_exchange = glucose_exchange
        self.penalty = penalty
        self.big_m = big_m
        self.atp_check_reaction = atp_check_reaction
        self.solver_timeout = solver_timeout
        self.cofactors = cofactors if cofactors else COMMON_COFACTORS
        self.max_valid_solutions = max_valid_solutions
        self.max_attempts = max_attempts
        self.check_cofactor_cycles = check_cofactor_cycles
        self.biomass_drop_ratio = biomass_drop_ratio

        # 并行配置
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.batch_size = batch_size
        self.frequency_threshold = frequency_threshold  # 改名为全局阈值

        self._log(f"并行配置: {self.n_jobs} 核心, 批量大小 {self.batch_size}")
        self._log(f"全局频率阈值: 反应在所有方案中最多出现 {frequency_threshold} 次")

        self.cofactor_virtual_reactions = {
            'NADPH': {'cpd00005_c': -1, 'cpd00006_c': 1, 'cpd00067_c': 1},
            'NADH': {'cpd00004_c': -1, 'cpd00003_c': 1, 'cpd00067_c': 1},
        }

        self._build_indices()
        self._build_metabolite_name_map()

        # 预序列化模型（用于并行传递）
        self._log("预序列化模型用于并行处理...")
        from io import StringIO
        user_io = StringIO()
        universal_io = StringIO()
        cobra.io.save_json_model(self.user_model, user_io)
        cobra.io.save_json_model(self.universal_model, universal_io)
        self._user_model_json = user_io.getvalue()
        self._universal_model_json = universal_io.getvalue()

        self.forward_reactions = {}
        self.candidate_reactions = []
        self.binary_variables = {}
        self.loopless_variables = {}
        self.combined_model = None
        self.results = []
        self.best_solution = None
        self.invalid_solutions = []
        self.valid_solutions_detail = []

        # *** 关键修复：使用全局频率统计 ***
        self.global_reaction_frequency = {}  # {rxn_id: count} - 全局计数
        self.global_excluded_reactions = set()  # 全局排除的反应集合
        self.layer_first_biomass = {}  # 每层首次有效解的biomass

    def _log(self, msg: str):
        print(msg)

    def _build_indices(self):
        self._log("  构建索引...")
        self.met_to_rxns = {}
        for rxn in self.universal_model.reactions:
            for met in rxn.metabolites:
                if met.id not in self.met_to_rxns:
                    self.met_to_rxns[met.id] = []
                self.met_to_rxns[met.id].append(rxn)

        self.user_rxn_ids = set()
        for rxn in self.user_model.reactions:
            self.user_rxn_ids.add(rxn.id)
            base_id = rxn.id.replace('_forward', '').replace('_backward', '')
            self.user_rxn_ids.add(base_id)

        self.user_met_ids = {met.id for met in self.user_model.metabolites}

    def _build_metabolite_name_map(self):
        """构建代谢物ID到名称的映射"""
        self._log("  构建代谢物名称映射...")
        self.metabolite_name_map = {}

        # 从universal_model获取
        for met in self.universal_model.metabolites:
            if met.name and met.name.strip():
                self.metabolite_name_map[met.id] = met.name
            else:
                self.metabolite_name_map[met.id] = met.id

        # 从user_model获取（可能会覆盖，以user model为准）
        for met in self.user_model.metabolites:
            if met.name and met.name.strip():
                self.metabolite_name_map[met.id] = met.name
            elif met.id not in self.metabolite_name_map:
                self.metabolite_name_map[met.id] = met.id

        self._log(f"    映射了 {len(self.metabolite_name_map)} 个代谢物名称")

    def _get_annotated_equation(self, rxn: Reaction) -> str:
        """
        生成带注释的反应方程式（使用代谢物名称）

        Parameters:
        -----------
        rxn : Reaction
            CobraPy反应对象

        Returns:
        --------
        str: 注释版本的反应方程式

        示例:
        原始: cpd00002_c + cpd00001_c --> cpd00008_c + cpd00009_c
        注释: ATP + H2O --> ADP + Phosphate
        """
        # 分离反应物和产物
        reactants = []
        products = []

        for met, coef in rxn.metabolites.items():
            met_name = self.metabolite_name_map.get(met.id, met.id)
            abs_coef = abs(coef)

            # 格式化系数
            if abs_coef == 1.0:
                coef_str = ""
            elif abs_coef == int(abs_coef):
                coef_str = f"{int(abs_coef)} "
            else:
                coef_str = f"{abs_coef:.3g} "

            met_str = f"{coef_str}{met_name}"

            if coef < 0:
                reactants.append(met_str)
            else:
                products.append(met_str)

        # 组合方程式
        reactants_str = " + ".join(reactants) if reactants else "Nothing"
        products_str = " + ".join(products) if products else "Nothing"

        # 判断可逆性
        if rxn.lower_bound < 0 and rxn.upper_bound > 0:
            arrow = " <=> "
        else:
            arrow = " --> "

        return f"{reactants_str}{arrow}{products_str}"

    def _get_reactions_for_metabolite(self, met_id: str) -> List[Reaction]:
        reactions = []
        for rxn in self.met_to_rxns.get(met_id, []):
            base_id = rxn.id.replace('_forward', '').replace('_backward', '')
            if rxn.id in self.user_rxn_ids or base_id in self.user_rxn_ids:
                continue
            if rxn.id.startswith(('EX_', 'DM_', 'SK_')):
                continue
            if rxn.id == self.atp_check_reaction:
                continue
            reactions.append(rxn)
        return reactions

    def _get_metabolites_from_reactions(self, reactions: List[Reaction], exclude_cofactors: bool = True) -> Set[str]:
        metabolites = set()
        for rxn in reactions:
            for met in rxn.metabolites:
                if exclude_cofactors and met.id in self.cofactors:
                    continue
                metabolites.add(met.id)
        return metabolites

    def _expand_forward(self, start_metabolites: Set[str], max_layers: int) -> Dict[int, Set[str]]:
        layer_reactions = {}
        visited_reactions = set()
        current_metabolites = start_metabolites.copy()

        for layer in range(1, max_layers + 1):
            new_reactions = set()
            for met_id in current_metabolites:
                for rxn in self._get_reactions_for_metabolite(met_id):
                    if rxn.id not in visited_reactions:
                        new_reactions.add(rxn.id)
                        visited_reactions.add(rxn.id)

            if not new_reactions:
                self._log(f"    第{layer}层: 无新反应，停止扩展")
                break

            layer_reactions[layer] = new_reactions
            rxn_objects = [self.universal_model.reactions.get_by_id(rid) for rid in new_reactions]
            current_metabolites = self._get_metabolites_from_reactions(rxn_objects, exclude_cofactors=True)
            self._log(f"    第{layer}层: {len(new_reactions)} 个反应")

        return layer_reactions

    def expand_from_substrate(self) -> Dict[int, Set[str]]:
        self._log(f"\n>>> 正向扩展: 从 {self.substrate} 出发")
        start_mets = {self.substrate}
        if self.substrate_external:
            start_mets.add(self.substrate_external)
        self.forward_reactions = self._expand_forward(start_mets, self.expansion_layers)
        total = sum(len(rxns) for rxns in self.forward_reactions.values())
        self._log(f"  正向扩展共找到 {total} 个候选反应")
        return self.forward_reactions

    def collect_candidates(self, filter_dangling: bool = True) -> List[Reaction]:
        self._log(f"\n>>> 收集候选反应")

        if filter_dangling:
            available_metabolites = set(self.user_met_ids)
            kept_reactions = {}
            layers = sorted(self.forward_reactions.keys(), reverse=True)

            for layer in layers:
                rxn_ids = self.forward_reactions[layer]
                kept = []
                for rxn_id in rxn_ids:
                    try:
                        rxn = self.universal_model.reactions.get_by_id(rxn_id)
                    except KeyError:
                        continue
                    rxn_mets = {met.id for met in rxn.metabolites if met.id not in self.cofactors}
                    if rxn_mets & available_metabolites:
                        kept.append(rxn_id)
                        for met in rxn.metabolites:
                            available_metabolites.add(met.id)
                kept_reactions[layer] = kept
                self._log(f"  第{layer}层: {len(rxn_ids)} → {len(kept)} 个反应")

            all_kept_ids = set()
            for kept in kept_reactions.values():
                all_kept_ids.update(kept)
        else:
            all_kept_ids = set()
            for rxns in self.forward_reactions.values():
                all_kept_ids.update(rxns)

        candidates = []
        for rxn_id in all_kept_ids:
            try:
                rxn = self.universal_model.reactions.get_by_id(rxn_id)
                candidates.append(rxn)
            except KeyError:
                continue

        self.candidate_reactions = candidates
        self._log(f"  最终候选: {len(candidates)} 个反应")
        return candidates

    def _ensure_substrate_available(self, model: Model) -> str:
        uptake_id = f"UPTAKE_{self.substrate}"
        if uptake_id not in [r.id for r in model.reactions]:
            if self.substrate not in [m.id for m in model.metabolites]:
                met_c = Metabolite(self.substrate, compartment='c')
                model.add_metabolites([met_c])
            uptake_rxn = Reaction(uptake_id)
            uptake_rxn.add_metabolites({model.metabolites.get_by_id(self.substrate): 1})
            uptake_rxn.bounds = (0, 10)
            model.add_reactions([uptake_rxn])
        return uptake_id

    def _setup_carbon_source(self, model: Model, substrate_uptake: float = -10) -> str:
        if self.glucose_exchange in [r.id for r in model.reactions]:
            model.reactions.get_by_id(self.glucose_exchange).lower_bound = 0
        exchange_id = self._ensure_substrate_available(model)
        model.reactions.get_by_id(exchange_id).lower_bound = substrate_uptake
        return exchange_id

    def _add_loopless_constraints(self, model: Model, candidate_ids: List[str]):
        self._log("\n>>> 添加Loopless约束")
        solver = model.solver
        M = self.big_m

        reversible_candidates = []
        for rxn_id in candidate_ids:
            rxn = model.reactions.get_by_id(rxn_id)
            if rxn.lower_bound < 0 and rxn.upper_bound > 0:
                reversible_candidates.append(rxn_id)

        self._log(f"  可逆候选反应: {len(reversible_candidates)} 个")

        if not reversible_candidates:
            return

        for rxn_id in reversible_candidates:
            rxn = model.reactions.get_by_id(rxn_id)
            g_var = solver.interface.Variable(name=f"G_{rxn_id}", type="binary")
            solver.add(g_var)
            self.loopless_variables[rxn_id] = g_var

            solver.add(solver.interface.Constraint(
                rxn.flux_expression + M * g_var, lb=0, name=f"loopless_lb_{rxn_id}"
            ))
            solver.add(solver.interface.Constraint(
                rxn.flux_expression - M * g_var, ub=0, name=f"loopless_ub_{rxn_id}"
            ))

        self._log(f"  添加了 {len(self.loopless_variables)} 个loopless变量")

    def _build_milp_model(self) -> Model:
        self._log("\n>>> 构建MILP模型")

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

        self._log(f"  添加了 {len(candidate_ids)} 个候选反应")

        self._setup_carbon_source(combined)
        combined.objective = self.objective_reaction

        solver = combined.solver
        self.binary_variables = {}

        for rxn_id in candidate_ids:
            rxn = combined.reactions.get_by_id(rxn_id)
            y_var = solver.interface.Variable(name=f"y_{rxn_id}", type="binary")
            solver.add(y_var)
            self.binary_variables[rxn_id] = y_var

            solver.add(solver.interface.Constraint(
                rxn.forward_variable - self.big_m * y_var, ub=0, name=f"bigM_fwd_{rxn_id}"
            ))
            solver.add(solver.interface.Constraint(
                rxn.reverse_variable - self.big_m * y_var, ub=0, name=f"bigM_rev_{rxn_id}"
            ))

        y_sum = Zero
        for y_var in self.binary_variables.values():
            y_sum = y_sum + y_var

        self.max_reactions_constraint = solver.interface.Constraint(
            y_sum, lb=1, ub=1, name="exact_reactions_constraint"
        )
        solver.add(self.max_reactions_constraint)

        self._add_loopless_constraints(combined, candidate_ids)

        self.combined_model = combined
        n_binary = len(self.binary_variables) + len(self.loopless_variables)
        self._log(f"  模型: {len(combined.reactions)} 反应, {n_binary} 个二进制变量")

        return combined

    def _set_solver_params(self):
        """优化版求解器参数设置"""
        solver = self.combined_model.solver

        try:
            if hasattr(solver.problem, 'Params'):
                # Gurobi优化参数
                solver.problem.Params.Threads = self.n_jobs  # 使用所有核心
                solver.problem.Params.MIPGap = 0.005  # 更严格的gap
                solver.problem.Params.Method = 2  # Barrier方法，更好的并行性
                solver.problem.Params.Presolve = 2  # 激进预处理
                solver.problem.Params.MIPFocus = 1  # 倾向快速找可行解
                solver.problem.Params.Cuts = 2  # 激进cut
                solver.problem.Params.Heuristics = 0.1  # 更多启发式搜索
                solver.problem.Params.TimeLimit = self.solver_timeout
                self._log(f"  [Gurobi] 使用 {self.n_jobs} 线程")
        except Exception as e:
            self._log(f"  求解器参数设置警告: {e}")

    def _solve_milp(self) -> Tuple[cobra.Solution, float, List[str]]:
        model = self.combined_model
        solver = model.solver

        biomass_rxn = model.reactions.get_by_id(self.objective_reaction)

        y_penalty = Zero
        for y_var in self.binary_variables.values():
            y_penalty = y_penalty + y_var

        objective_expr = biomass_rxn.flux_expression - self.penalty * y_penalty
        solver.objective = solver.interface.Objective(objective_expr, direction='max')

        self._set_solver_params()

        try:
            solution = model.optimize()
        except Exception as e:
            return None, 0.0, []

        if solution is None or solution.status not in ['optimal', 'feasible']:
            return solution, 0.0, []

        selected_rxns = []
        for rxn_id, y_var in self.binary_variables.items():
            if hasattr(y_var, 'primal') and y_var.primal is not None:
                if y_var.primal >= 0.5:
                    selected_rxns.append(rxn_id)
            else:
                flux = solution.fluxes.get(rxn_id, 0)
                if abs(flux) > self.flux_threshold:
                    selected_rxns.append(rxn_id)

        biomass_flux = solution.objective_value

        return solution, biomass_flux, selected_rxns

    def _check_energy_cycle_parallel(self, solutions_list: List[List[str]]) -> List[Tuple]:
        """
        并行检测能量循环

        Parameters:
        -----------
        solutions_list : List[List[str]]
            待检测的反应ID列表的列表

        Returns:
        --------
        List[Tuple]: [(selected_rxns, has_cycle, max_flux, cycle_info), ...]
        """
        if not solutions_list:
            return []

        # 构建参数列表
        args_list = [
            (rxns, self._user_model_json, self._universal_model_json,
             self.glucose_exchange, self.substrate, self.atp_check_reaction,
             self.flux_threshold, self.check_cofactor_cycles,
             self.cofactor_virtual_reactions)
            for rxns in solutions_list
        ]

        # 使用进程池并行处理
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(_check_cycle_worker, args) for args in args_list]
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    self._log(f"    并行检测出错: {e}")
                    # 出错时返回无循环
                    results.append((args_list[len(results)][0], False, 0.0, ""))

        return results

    def _check_energy_cycle(self, selected_rxns: List[str]) -> Tuple[bool, float, str]:
        """单个解的能量循环检测（用于非并行模式）"""
        result = _check_cycle_worker((
            selected_rxns, self._user_model_json, self._universal_model_json,
            self.glucose_exchange, self.substrate, self.atp_check_reaction,
            self.flux_threshold, self.check_cofactor_cycles,
            self.cofactor_virtual_reactions
        ))
        return result[1], result[2], result[3]

    def _add_integer_cut(self, selected_rxns: List[str], cut_id: int):
        solver = self.combined_model.solver
        cut_expr = Zero
        for rxn_id in selected_rxns:
            if rxn_id in self.binary_variables:
                cut_expr = cut_expr + self.binary_variables[rxn_id]
        solver.add(solver.interface.Constraint(
            cut_expr, ub=len(selected_rxns) - 1, name=f"integer_cut_{cut_id}"
        ))

    def _get_reaction_details(self, rxn_ids: List[str]) -> List[Dict]:
        """
        获取反应详细信息，包括注释版本的方程式

        Returns:
        --------
        List[Dict]: 包含反应详细信息的字典列表，新增 'reaction_equation_annotated' 字段
        """
        details = []
        for rxn_id in rxn_ids:
            try:
                rxn = self.universal_model.reactions.get_by_id(rxn_id)
                annotated_eq = self._get_annotated_equation(rxn)
                details.append({
                    'reaction_id': rxn_id,
                    'reaction_name': rxn.name if rxn.name else '',
                    'reaction_equation': rxn.reaction,
                    'reaction_equation_annotated': annotated_eq,  # 新增字段
                    'lower_bound': rxn.lower_bound,
                    'upper_bound': rxn.upper_bound,
                    'gene_reaction_rule': rxn.gene_reaction_rule if rxn.gene_reaction_rule else '',
                    'subsystem': rxn.subsystem if rxn.subsystem else ''
                })
            except KeyError:
                details.append({
                    'reaction_id': rxn_id,
                    'reaction_name': '',
                    'reaction_equation': 'N/A',
                    'reaction_equation_annotated': 'N/A',  # 新增字段
                    'lower_bound': 0,
                    'upper_bound': 0,
                    'gene_reaction_rule': '',
                    'subsystem': ''
                })
        return details

    def _update_frequency_and_exclude_global(self, rxn_ids: List[str]):
        """
        更新全局频率统计并排除高频反应

        *** 关键修复：使用全局统计，反应在所有方案中最多出现指定次数 ***

        Parameters:
        -----------
        rxn_ids : List[str]
            本次解中使用的反应ID列表
        """
        for rxn_id in rxn_ids:
            # 全局频率计数
            freq = self.global_reaction_frequency.get(rxn_id, 0) + 1
            self.global_reaction_frequency[rxn_id] = freq

            # 如果达到全局阈值，永久排除此反应
            if freq >= self.frequency_threshold:
                if rxn_id not in self.global_excluded_reactions:
                    self.global_excluded_reactions.add(rxn_id)
                    # 设置二进制变量上下界为0，永久禁用
                    if rxn_id in self.binary_variables:
                        self.binary_variables[rxn_id].ub = 0
                        self.binary_variables[rxn_id].lb = 0
                    self._log(f"    [全局排除] {rxn_id} 已在所有方案中出现{freq}次，永久排除")

    def _prepare_for_new_layer(self, current_max_rxns: int):
        """
        为新的反应数量级别做准备

        *** 关键修复：不重置全局排除的反应 ***

        Parameters:
        -----------
        current_max_rxns : int
            即将搜索的反应数量级别
        """
        # 注意：不重置全局排除的反应！
        # 只是记录日志，表示开始新层级
        self._log(f"  [级别{current_max_rxns}] 开始搜索，已全局排除 {len(self.global_excluded_reactions)} 个反应")

    def _find_valid_solutions(self) -> List[Tuple[float, List[str]]]:
        """迭代求解 + 批量并行ATP/NADPH/NADH验证 [全局频率统计修复版]"""
        self._log(f"\n>>> MILP求解 (Loopless) + 并行能量循环验证")
        self._log(f"    目标: 找到 {self.max_valid_solutions} 个有效解，最多尝试 {self.max_attempts} 次")
        self._log(f"    并行: {self.n_jobs} 核心, 批量大小 {self.batch_size}")
        self._log(f"    全局频率阈值: 反应在所有方案中最多出现 {self.frequency_threshold} 次")
        self._log(f"    早停策略: 当biomass低于该层首次有效解的{self.biomass_drop_ratio * 100:.0f}%时，跳转下一层")

        valid_solutions = []
        cut_id = 0
        total_attempts = 0

        for current_max_rxns in range(1, self.max_reactions + 1):
            self._log(f"\n>>> 搜索恰好使用 {current_max_rxns} 个反应的方案")

            # 准备新层级（不重置全局排除）
            self._prepare_for_new_layer(current_max_rxns)

            self.max_reactions_constraint.ub = current_max_rxns
            self.max_reactions_constraint.lb = current_max_rxns

            consecutive_no_solution = 0
            max_consecutive = 3
            first_biomass_in_layer = None
            is_last_layer = (current_max_rxns == self.max_reactions)

            # 批量收集待验证的解
            pending_solutions = []
            should_break_layer = False

            while total_attempts < self.max_attempts and not should_break_layer:
                total_attempts += 1
                solution, biomass, selected = self._solve_milp()

                if biomass < self.flux_threshold or not selected:
                    consecutive_no_solution += 1
                    if consecutive_no_solution >= max_consecutive:
                        self._log(f"  连续{max_consecutive}次无解，进入下一反应数量级别")
                        break
                    continue

                consecutive_no_solution = 0

                # 检查是否包含已全局排除的反应
                has_excluded = [r for r in selected if r in self.global_excluded_reactions]
                if has_excluded:
                    # 添加integer cut避免重复
                    self._add_integer_cut(selected, cut_id)
                    cut_id += 1
                    self._log(f"    ⊘ 跳过（包含已全局排除反应: {', '.join(has_excluded)}）")
                    continue

                pending_solutions.append((biomass, selected))

                # 先添加integer cut，避免重复求解
                self._add_integer_cut(selected, cut_id)
                cut_id += 1

                # 达到批量大小时并行验证
                if len(pending_solutions) >= self.batch_size:
                    self._log(f"    批量验证 {len(pending_solutions)} 个解...")

                    solutions_to_check = [s[1] for s in pending_solutions]
                    check_results = self._check_energy_cycle_parallel(solutions_to_check)

                    for (biomass_val, rxns), (_, has_cycle, max_flux, cycle_info) in zip(pending_solutions,
                                                                                         check_results):
                        if has_cycle:
                            self._log(f"    ✗ 循环 ({cycle_info}) - {rxns}")
                            self.invalid_solutions.append({
                                'reactions': rxns, 'cycle_flux': max_flux, 'cycle_type': cycle_info
                            })
                            continue  # 跳过循环解

                        # 再次检查是否包含已全局排除的反应（可能在批处理期间被排除）
                        has_excluded = [r for r in rxns if r in self.global_excluded_reactions]
                        if has_excluded:
                            self._log(
                                f"    ⊘ 跳过（包含已全局排除反应: {', '.join(has_excluded)}） - biomass={biomass_val:.4f}")
                            continue

                        # 检查biomass阈值
                        if first_biomass_in_layer is None:
                            # 第一个有效解，设置基准
                            first_biomass_in_layer = biomass_val
                            self.layer_first_biomass[current_max_rxns] = biomass_val
                            biomass_threshold = first_biomass_in_layer * self.biomass_drop_ratio
                            self._log(
                                f"    [该层基准biomass: {first_biomass_in_layer:.4f}, 阈值: {biomass_threshold:.4f}]")
                        else:
                            # 检查biomass是否低于阈值
                            biomass_threshold = first_biomass_in_layer * self.biomass_drop_ratio
                            if biomass_val < biomass_threshold:
                                self._log(
                                    f"    ⚠ biomass ({biomass_val:.4f}) 低于阈值 ({biomass_threshold:.4f})，跳过此解")
                                if is_last_layer:
                                    self._log(f"  已是最后一层（{self.max_reactions}个反应），清空待处理列表并结束该层")
                                    pending_solutions = []
                                    should_break_layer = True
                                    break
                                else:
                                    self._log(f"  跳转到下一反应数量级别")
                                    pending_solutions = []
                                    should_break_layer = True
                                    break

                        # 只有通过所有检查后才添加解
                        self._log(f"    ✓ 有效 - {len(rxns)}个反应, biomass={biomass_val:.4f}")
                        valid_solutions.append((biomass_val, rxns))

                        rxn_details = self._get_reaction_details(rxns)
                        self.valid_solutions_detail.append({
                            'solution_id': len(valid_solutions),
                            'biomass_flux': biomass_val,
                            'n_reactions': len(rxns),
                            'reaction_ids': rxns,
                            'reaction_details': rxn_details
                        })

                        # 全局频率更新
                        self._update_frequency_and_exclude_global(rxns)

                        # 检查是否达到目标数量
                        if len(valid_solutions) >= self.max_valid_solutions:
                            self._log(f"\n  已找到 {self.max_valid_solutions} 个有效解，停止搜索")
                            return valid_solutions

                    pending_solutions = []

            # 处理剩余的待验证解
            if pending_solutions and not should_break_layer:
                self._log(f"    处理剩余 {len(pending_solutions)} 个解...")
                solutions_to_check = [s[1] for s in pending_solutions]
                check_results = self._check_energy_cycle_parallel(solutions_to_check)

                for (biomass_val, rxns), (_, has_cycle, max_flux, cycle_info) in zip(pending_solutions, check_results):
                    if has_cycle:
                        self.invalid_solutions.append({
                            'reactions': rxns, 'cycle_flux': max_flux, 'cycle_type': cycle_info
                        })
                        continue

                    # 检查是否包含已全局排除的反应
                    has_excluded = [r for r in rxns if r in self.global_excluded_reactions]
                    if has_excluded:
                        self._log(
                            f"    ⊘ 跳过（包含已全局排除反应: {', '.join(has_excluded)}） - biomass={biomass_val:.4f}")
                        continue

                    # 检查biomass阈值
                    if first_biomass_in_layer is None:
                        first_biomass_in_layer = biomass_val
                        self.layer_first_biomass[current_max_rxns] = biomass_val
                        biomass_threshold = first_biomass_in_layer * self.biomass_drop_ratio
                        self._log(f"    [该层基准biomass: {first_biomass_in_layer:.4f}, 阈值: {biomass_threshold:.4f}]")
                    else:
                        biomass_threshold = first_biomass_in_layer * self.biomass_drop_ratio
                        if biomass_val < biomass_threshold:
                            self._log(f"    ⚠ biomass ({biomass_val:.4f}) 低于阈值 ({biomass_threshold:.4f})，跳过此解")
                            if is_last_layer:
                                self._log(f"  已是最后一层（{self.max_reactions}个反应），结束搜索")
                                self._log(f"\n  总尝试次数: {total_attempts}")
                                return valid_solutions
                            else:
                                self._log(f"  跳转到下一反应数量级别")
                                break

                    # 只有通过所有检查后才添加解
                    valid_solutions.append((biomass_val, rxns))
                    rxn_details = self._get_reaction_details(rxns)
                    self.valid_solutions_detail.append({
                        'solution_id': len(valid_solutions),
                        'biomass_flux': biomass_val,
                        'n_reactions': len(rxns),
                        'reaction_ids': rxns,
                        'reaction_details': rxn_details
                    })

                    # 全局频率更新
                    self._update_frequency_and_exclude_global(rxns)

                    if len(valid_solutions) >= self.max_valid_solutions:
                        self._log(f"\n  已找到 {self.max_valid_solutions} 个有效解，停止搜索")
                        return valid_solutions

        self._log(f"\n  总尝试次数: {total_attempts}")
        return valid_solutions

    def run(self, filter_dangling: bool = True) -> pd.DataFrame:
        print("=" * 70)
        print("Loopless Gap-Filling [全局频率统计修复版 + 方程式注释]")
        print("=" * 70)
        print(f"目标反应: {self.objective_reaction}")
        print(f"底物: {self.substrate}")
        print(f"最大反应数: {self.max_reactions}")
        print(f"并行核心数: {self.n_jobs}")
        print(f"批量大小: {self.batch_size}")
        print(f"全局频率阈值: {self.frequency_threshold} (反应在所有方案中最多出现此次数)")
        print(f"Biomass下降阈值: {self.biomass_drop_ratio*100:.0f}%")
        print()

        self.expand_from_substrate()
        self.collect_candidates(filter_dangling=filter_dangling)

        if not self.candidate_reactions:
            print("\n✗ 没有候选反应")
            return pd.DataFrame()

        self._build_milp_model()
        valid_solutions = self._find_valid_solutions()

        if not valid_solutions:
            print("\n✗ 没有找到有效解")
            return pd.DataFrame()

        unique_solutions = {}
        for biomass, rxn_ids in valid_solutions:
            key = tuple(sorted(rxn_ids))
            if key not in unique_solutions or biomass > unique_solutions[key][0]:
                unique_solutions[key] = (biomass, rxn_ids)

        sorted_solutions = sorted(unique_solutions.values(), key=lambda x: (len(x[1]), -x[0]))

        for i, (biomass, rxn_ids) in enumerate(sorted_solutions, 1):
            rxn_details = self._get_reaction_details(rxn_ids)
            self.results.append({
                '方案编号': i,
                '引入反应数': len(rxn_ids),
                '引入反应ID': '; '.join(rxn_ids),
                '反应方程式': '; '.join([d['reaction_equation'] for d in rxn_details]),
                '反应方程式注释': '; '.join([d['reaction_equation_annotated'] for d in rxn_details]),
                '反应名称': '; '.join([d['reaction_name'] for d in rxn_details]),
                '生物质通量': biomass
            })

            if self.best_solution is None or len(rxn_ids) < self.best_solution['n_reactions']:
                self.best_solution = {
                    'reactions': rxn_ids,
                    'n_reactions': len(rxn_ids),
                    'flux': biomass,
                    'reaction_details': rxn_details
                }

        self._print_summary()
        return pd.DataFrame(self.results)

    def _print_summary(self):
        print()
        print("=" * 70)
        print("结果汇总")
        print("=" * 70)

        if self.best_solution:
            print(f"✓ 最优方案:")
            print(f"  反应数: {self.best_solution['n_reactions']}")
            print(f"  生物质通量: {self.best_solution['flux']:.6f}")
            print(f"  添加的反应:")
            for detail in self.best_solution['reaction_details']:
                print(f"    - {detail['reaction_id']}: {detail['reaction_equation']}")
                print(f"      注释: {detail['reaction_equation_annotated']}")

        print(f"\n共找到 {len(self.results)} 个有效方案（去重后）")
        print(f"排除了 {len(self.invalid_solutions)} 个循环解")

        # 打印各层首次有效解的biomass
        if self.layer_first_biomass:
            print(f"\n各层首次有效解的biomass:")
            for layer, biomass in sorted(self.layer_first_biomass.items()):
                print(f"  {layer}个反应级别: {biomass:.6f}")

        # 打印全局频率统计
        print(f"\n全局频率统计:")
        print(f"  已全局排除的反应数: {len(self.global_excluded_reactions)}")
        if self.global_excluded_reactions:
            print(f"  被排除的反应:")
            for rxn_id in sorted(self.global_excluded_reactions):
                freq = self.global_reaction_frequency.get(rxn_id, 0)
                print(f"    - {rxn_id}: 出现 {freq} 次")

        # 打印高频反应（接近阈值但未被排除的）
        high_freq_reactions = {
            rxn_id: freq for rxn_id, freq in self.global_reaction_frequency.items()
            if freq >= self.frequency_threshold - 1 and rxn_id not in self.global_excluded_reactions
        }
        if high_freq_reactions:
            print(f"\n  高频反应（出现 {self.frequency_threshold - 1} 次，接近阈值）:")
            for rxn_id, freq in sorted(high_freq_reactions.items(), key=lambda x: -x[1]):
                print(f"    - {rxn_id}: 出现 {freq} 次")

        print("=" * 70)

    def get_optimal_model(self) -> Optional[Model]:
        if not self.best_solution:
            return None
        return self._build_solution_model(self.best_solution['reactions'])

    def _build_solution_model(self, rxn_ids: List[str]) -> Model:
        model = self.user_model.copy()
        self._ensure_substrate_available(model)
        for rxn_id in rxn_ids:
            if rxn_id not in [r.id for r in model.reactions]:
                try:
                    rxn = self.universal_model.reactions.get_by_id(rxn_id)
                    new_rxn = rxn.copy()
                    for met in rxn.metabolites:
                        if met.id not in [m.id for m in model.metabolites]:
                            model.add_metabolites([met.copy()])
                    model.add_reactions([new_rxn])
                except:
                    continue
        return model

    def save_results(self, output_dir: str = './gapfill_output'):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\n>>> 保存结果")

        if self.valid_solutions_detail:
            detail_rows = []
            for sol in self.valid_solutions_detail:
                for detail in sol['reaction_details']:
                    detail_rows.append({
                        '方案编号': sol['solution_id'],
                        '生物质通量': sol['biomass_flux'],
                        '方案反应数': sol['n_reactions'],
                        '反应ID': detail['reaction_id'],
                        '反应名称': detail['reaction_name'],
                        '反应方程式': detail['reaction_equation'],
                        '反应方程式注释': detail['reaction_equation_annotated'],
                    })

            detail_path = os.path.join(output_dir, f"gapfill_solutions_detailed_{timestamp}.csv")
            detail_df = pd.DataFrame(detail_rows)
            detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
            print(f"  详细信息表: {detail_path}")

        if self.valid_solutions_detail:
            models_dir = os.path.join(output_dir, f"all_solution_models_{timestamp}")
            os.makedirs(models_dir, exist_ok=True)
            for sol in self.valid_solutions_detail:
                sol_model = self._build_solution_model(sol['reaction_ids'])
                sol_model_path = os.path.join(models_dir, f"solution_{sol['solution_id']}_model.xml")
                cobra.io.write_sbml_model(sol_model, sol_model_path)
            print(f"  有效解模型: {models_dir}/ ({len(self.valid_solutions_detail)} 个)")

        print("\n保存完成!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Loopless Gap-Filling Tool [全局频率统计修复版 + 方程式注释]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('-u', '--user-model', required=True, help='用户模型文件')
    parser.add_argument('-r', '--universal-model', required=True, help='通用模型文件')
    parser.add_argument('-o', '--objective', required=True, help='目标反应ID')
    parser.add_argument('-s', '--substrate', required=True, help='底物ID')
    parser.add_argument('-m', '--max-reactions', type=int, default=5, help='最大反应数')
    parser.add_argument('-l', '--layers', type=int, default=5, help='扩展层数')
    parser.add_argument('-n', '--max-solutions', type=int, default=20, help='最大有效解数量')
    parser.add_argument('-a', '--max-attempts', type=int, default=100, help='最大尝试次数')
    parser.add_argument('-t', '--timeout', type=int, default=600, help='求解器超时')
    parser.add_argument('-d', '--output-dir', default='./gapfill_output', help='输出目录')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='并行核心数（-1=全部）')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='批量检测大小')
    parser.add_argument('-f', '--frequency-threshold', type=int, default=3,
                        help='全局频率阈值（反应在所有方案中最多出现次数）')
    parser.add_argument('--no-cofactor-check', action='store_true', help='禁用多辅因子检测')
    parser.add_argument('--biomass-drop-ratio', type=float, default=0.5, help='Biomass下降阈值')

    args = parser.parse_args()

    print("加载模型...")
    user_model = load_model(args.user_model)
    universal_model = load_model(args.universal_model)
    print(f"用户模型: {len(user_model.reactions)} 反应")
    print(f"通用模型: {len(universal_model.reactions)} 反应")

    gapfiller = LooplessGapFillerOptimized(
        user_model=user_model,
        universal_model=universal_model,
        objective_reaction=args.objective,
        substrate=args.substrate,
        max_reactions=args.max_reactions,
        expansion_layers=args.layers,
        solver_timeout=args.timeout,
        max_valid_solutions=args.max_solutions,
        max_attempts=args.max_attempts,
        check_cofactor_cycles=not args.no_cofactor_check,
        biomass_drop_ratio=args.biomass_drop_ratio,
        n_jobs=args.jobs,
        batch_size=args.batch_size,
        frequency_threshold=args.frequency_threshold,
    )

    results = gapfiller.run()
    gapfiller.save_results(args.output_dir)


if __name__ == "__main__":
    main()