#!/usr/bin/env python3
"""
개선된 평가 방법론

Church 논문에서 지적한 Word2Vec 평가의 문제점들을 해결하는 
더 신뢰할 수 있는 평가 방법을 구현합니다.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import json

@dataclass
class AnalogyQuestion:
    """아날로지 질문을 나타내는 클래스"""
    goal_a: str
    tactic_a: str
    goal_b: str
    correct_tactic: str
    category: str  # 문제 유형 (예: "arithmetic", "logic", "set_theory")
    difficulty: str  # 난이도 (easy, medium, hard)

class ImprovedEvaluator:
    """
    Church 논문의 문제점을 해결한 개선된 평가기
    """
    
    def __init__(self):
        self.test_questions = []
        self.vocabulary_stats = {}
        
    def create_diverse_test_set(self) -> List[AnalogyQuestion]:
        """
        논문에서 지적한 문제를 피하는 다양한 테스트 세트 생성
        
        Church 논문의 문제점:
        1. 모든 단어가 여러 위치에 나타남 (중복 문제)
        2. 테스트가 쉽게 "게임"될 수 있음
        3. 실제 SAT 문제와 차이가 큼
        """
        questions = []
        
        # 카테고리별로 분리된 질문들 생성
        
        # 1. 산술 연산 아날로지
        arithmetic_questions = [
            AnalogyQuestion(
                "∀ n : ℕ, n + 0 = n", "simp",
                "∀ x : ℝ, x + 0 = x", "simp",
                "arithmetic", "easy"
            ),
            AnalogyQuestion(
                "∀ a b : ℕ, a + b = b + a", "ring",
                "∀ a b : ℕ, a * b = b * a", "ring",
                "arithmetic", "medium"
            ),
            AnalogyQuestion(
                "∀ n : ℕ, n * 1 = n", "simp",
                "∀ x : ℝ, x * 1 = x", "simp",
                "arithmetic", "easy"
            ),
        ]
        
        # 2. 논리 연산 아날로지
        logic_questions = [
            AnalogyQuestion(
                "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P", "simp",
                "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P", "simp",
                "logic", "medium"
            ),
            AnalogyQuestion(
                "∀ P : Prop, P ∧ True ↔ P", "simp",
                "∀ P : Prop, P ∨ False ↔ P", "simp",
                "logic", "easy"
            ),
        ]
        
        # 3. 집합론 아날로지
        set_theory_questions = [
            AnalogyQuestion(
                "∀ s : Set α, s ∪ ∅ = s", "simp",
                "∀ s : Set α, s ∩ univ = s", "simp",
                "set_theory", "medium"
            ),
        ]
        
        # 4. 리스트 연산 아날로지
        list_questions = [
            AnalogyQuestion(
                "∀ l : List α, l ++ [] = l", "simp",
                "∀ l : List α, [] ++ l = l", "simp",
                "list", "easy"
            ),
        ]
        
        questions.extend(arithmetic_questions)
        questions.extend(logic_questions) 
        questions.extend(set_theory_questions)
        questions.extend(list_questions)
        
        return questions
    
    def analyze_vocabulary_overlap(self, questions: List[AnalogyQuestion]) -> Dict:
        """
        Church 논문에서 언급한 어휘 중복 문제를 분석합니다.
        
        SAT 문제는 거의 중복이 없어야 하는데, questions-words는 
        모든 단어가 여러 위치에 나타나는 문제가 있었습니다.
        """
        vocab_positions = defaultdict(lambda: {"goal_a": 0, "tactic_a": 0, "goal_b": 0, "correct_tactic": 0})
        
        for q in questions:
            # 간단한 토큰화 (실제로는 더 정교한 토큰화 필요)
            goal_a_tokens = q.goal_a.split()
            goal_b_tokens = q.goal_b.split()
            
            for token in goal_a_tokens:
                vocab_positions[token]["goal_a"] += 1
            for token in goal_b_tokens:
                vocab_positions[token]["goal_b"] += 1
            
            vocab_positions[q.tactic_a]["tactic_a"] += 1
            vocab_positions[q.correct_tactic]["correct_tactic"] += 1
        
        # 중복 분석
        overlap_stats = {
            "total_unique_words": len(vocab_positions),
            "words_in_multiple_positions": 0,
            "overlap_patterns": []
        }
        
        for word, positions in vocab_positions.items():
            num_positions = sum(1 for count in positions.values() if count > 0)
            if num_positions > 1:
                overlap_stats["words_in_multiple_positions"] += 1
                overlap_stats["overlap_patterns"].append({
                    "word": word,
                    "positions": {k: v for k, v in positions.items() if v > 0}
                })
        
        return overlap_stats
    
    def evaluate_with_multiple_metrics(self, trainer, questions: List[AnalogyQuestion]) -> Dict:
        """
        논문에서 제안한 여러 평가 지표를 사용한 종합 평가
        """
        results = {
            "accuracy_by_method": {},
            "accuracy_by_category": {},
            "accuracy_by_difficulty": {},
            "top_k_accuracy": {},
            "detailed_results": []
        }
        
        # 여러 아날로지 방법들 테스트
        methods = ["Original_Word2Vec", "Levy2014a_Sum", "Levy2014b_Mult"]
        
        for method in methods:
            results["accuracy_by_method"][method] = {"correct": 0, "total": 0}
            results["top_k_accuracy"][method] = {"top1": 0, "top3": 0, "top5": 0}
        
        # 카테고리/난이도별 정확도 초기화
        categories = set(q.category for q in questions)
        difficulties = set(q.difficulty for q in questions)
        
        for cat in categories:
            results["accuracy_by_category"][cat] = {method: {"correct": 0, "total": 0} for method in methods}
        for diff in difficulties:
            results["accuracy_by_difficulty"][diff] = {method: {"correct": 0, "total": 0} for method in methods}
        
        # 각 질문에 대해 평가
        for question in questions:
            question_result = {
                "question": {
                    "goal_a": question.goal_a,
                    "tactic_a": question.tactic_a, 
                    "goal_b": question.goal_b,
                    "correct_answer": question.correct_tactic,
                    "category": question.category,
                    "difficulty": question.difficulty
                },
                "predictions": {}
            }
            
            try:
                # 여러 방법으로 예측
                all_results = trainer.compare_analogy_methods(
                    question.goal_a, question.tactic_a, question.goal_b, top_k=5
                )
                
                for method, predictions in all_results.items():
                    question_result["predictions"][method] = predictions
                    
                    # 정확도 계산
                    results["accuracy_by_method"][method]["total"] += 1
                    results["accuracy_by_category"][question.category][method]["total"] += 1
                    results["accuracy_by_difficulty"][question.difficulty][method]["total"] += 1
                    
                    # Top-K 정확도
                    pred_tactics = [pred[0] for pred in predictions]
                    if question.correct_tactic in pred_tactics[:1]:
                        results["top_k_accuracy"][method]["top1"] += 1
                        results["accuracy_by_method"][method]["correct"] += 1
                        results["accuracy_by_category"][question.category][method]["correct"] += 1
                        results["accuracy_by_difficulty"][question.difficulty][method]["correct"] += 1
                    if question.correct_tactic in pred_tactics[:3]:
                        results["top_k_accuracy"][method]["top3"] += 1
                    if question.correct_tactic in pred_tactics[:5]:
                        results["top_k_accuracy"][method]["top5"] += 1
                        
            except Exception as e:
                question_result["error"] = str(e)
            
            results["detailed_results"].append(question_result)
        
        # 최종 정확도 계산
        for method in methods:
            total = results["accuracy_by_method"][method]["total"]
            if total > 0:
                results["accuracy_by_method"][method]["percentage"] = \
                    results["accuracy_by_method"][method]["correct"] / total * 100
                
                for k in ["top1", "top3", "top5"]:
                    results["top_k_accuracy"][method][f"{k}_percentage"] = \
                        results["top_k_accuracy"][method][k] / total * 100
        
        return results
    
    def generate_evaluation_report(self, trainer, save_path: str = "evaluation_report.json"):
        """
        종합적인 평가 리포트를 생성합니다.
        """
        print("=== 개선된 평가 방법론 실행 ===\n")
        
        print("1. 다양한 테스트 세트 생성...")
        questions = self.create_diverse_test_set()
        print(f"   생성된 질문 수: {len(questions)}")
        
        print("2. 어휘 중복 분석...")
        vocab_analysis = self.analyze_vocabulary_overlap(questions)
        print(f"   총 고유 단어 수: {vocab_analysis['total_unique_words']}")
        print(f"   여러 위치에 나타나는 단어 수: {vocab_analysis['words_in_multiple_positions']}")
        
        if vocab_analysis['words_in_multiple_positions'] > 0:
            print("   ⚠️  중복 패턴 발견 - Church 논문에서 지적한 문제 존재")
            for pattern in vocab_analysis['overlap_patterns'][:3]:  # 처음 3개만 표시
                print(f"     '{pattern['word']}': {pattern['positions']}")
        else:
            print("   ✓ 중복 없음 - SAT 스타일 테스트")
        
        print("\n3. 다중 지표 평가 실행...")
        evaluation_results = self.evaluate_with_multiple_metrics(trainer, questions)
        
        print("\n4. 평가 결과:")
        for method, results in evaluation_results["accuracy_by_method"].items():
            if "percentage" in results:
                print(f"   {method}: {results['percentage']:.1f}% 정확도")
        
        print("\n5. Top-K 정확도:")
        for method in evaluation_results["top_k_accuracy"]:
            topk = evaluation_results["top_k_accuracy"][method]
            if "top1_percentage" in topk:
                print(f"   {method}:")
                print(f"     Top-1: {topk['top1_percentage']:.1f}%")
                print(f"     Top-3: {topk['top3_percentage']:.1f}%")
                print(f"     Top-5: {topk['top5_percentage']:.1f}%")
        
        # 리포트 저장
        full_report = {
            "vocabulary_analysis": vocab_analysis,
            "evaluation_results": evaluation_results,
            "test_questions": [
                {
                    "goal_a": q.goal_a,
                    "tactic_a": q.tactic_a,
                    "goal_b": q.goal_b,
                    "correct_tactic": q.correct_tactic,
                    "category": q.category,
                    "difficulty": q.difficulty
                } for q in questions
            ]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 평가 리포트 저장: {save_path}")
        print("\n=== 평가 완료 ===")
        
        return full_report

def main():
    """메인 함수 - 단독 실행용"""
    print("개선된 평가 방법론 테스트를 위해서는")
    print("trained Goal2Vec 모델이 필요합니다.")
    print()
    print("사용법:")
    print("from improved_evaluation import ImprovedEvaluator")
    print("evaluator = ImprovedEvaluator()")
    print("evaluator.generate_evaluation_report(trainer)")

if __name__ == "__main__":
    main()