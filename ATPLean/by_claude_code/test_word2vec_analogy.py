#!/usr/bin/env python3
"""
Word2Vec 스타일 아날로지 기능 테스트 스크립트

논문에서 언급한 다양한 아날로지 방법들을 비교 테스트합니다.
"""

from goal2vec_model import Goal2VecTrainer, EmbeddingConfig
from minif2f_processor import MathProblem, ProblemDifficulty

def create_math_analogy_problems():
    """수학적 아날로지 테스트를 위한 문제들을 생성합니다."""
    problems = [
        # 자연수 기본 연산
        MathProblem("nat_add_zero", "∀ n : ℕ, n + 0 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_zero_add", "∀ n : ℕ, 0 + n = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_add_comm", "∀ a b : ℕ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("nat_mul_zero", "∀ n : ℕ, n * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_mul_one", "∀ n : ℕ, n * 1 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_mul_comm", "∀ a b : ℕ, a * b = b * a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # 실수 연산
        MathProblem("real_add_zero", "∀ x : ℝ, x + 0 = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_zero_add", "∀ x : ℝ, 0 + x = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_add_comm", "∀ a b : ℝ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("real_mul_zero", "∀ x : ℝ, x * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_mul_one", "∀ x : ℝ, x * 1 = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_mul_comm", "∀ a b : ℝ, a * b = b * a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # 논리 연산
        MathProblem("and_comm", "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P", proof="by simp [and_comm]", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("or_comm", "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P", proof="by simp [or_comm]", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("and_true", "∀ P : Prop, P ∧ True ↔ P", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("or_false", "∀ P : Prop, P ∨ False ↔ P", proof="by simp", difficulty=ProblemDifficulty.EASY),
        
        # 리스트 연산
        MathProblem("list_nil_append", "∀ l : List α, l ++ [] = l", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("list_append_nil", "∀ l : List α, [] ++ l = l", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("list_append_assoc", "∀ a b c : List α, (a ++ b) ++ c = a ++ (b ++ c)", proof="by simp [List.append_assoc]", difficulty=ProblemDifficulty.MEDIUM),
    ]
    return problems

def test_analogy_methods():
    """다양한 아날로지 방법들을 테스트합니다."""
    print("=== Word2Vec 스타일 아날로지 기능 테스트 ===\n")
    
    # 설정
    config = EmbeddingConfig(
        embedding_dim=128,
        epochs=30,
        batch_size=16,
        learning_rate=0.01,
        min_count=1
    )
    
    print("1. 트레이닝 데이터 생성...")
    problems = create_math_analogy_problems()
    print(f"   생성된 문제 수: {len(problems)}")
    
    print("2. Goal2Vec 모델 초기화 및 훈련...")
    trainer = Goal2VecTrainer(config)
    trainer.prepare_training_data(problems)
    trainer.train()
    print("   ✓ 모델 훈련 완료")
    
    print("\n3. 아날로지 테스트 시작...")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "자연수 → 실수 아날로지",
            "goal_a": "∀ n : ℕ, n + 0 = n",
            "tactic_a": "simp",
            "goal_b": "∀ x : ℝ, x + 0 = x",
            "expected": "simp"
        },
        {
            "name": "덧셈 → 곱셈 아날로지",
            "goal_a": "∀ a b : ℕ, a + b = b + a",
            "tactic_a": "ring",
            "goal_b": "∀ a b : ℕ, a * b = b * a",
            "expected": "ring"
        },
        {
            "name": "AND → OR 논리 아날로지",
            "goal_a": "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P",
            "tactic_a": "simp",
            "goal_b": "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P",
            "expected": "simp"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i}: {test_case['name']} ---")
        print(f"입력: {test_case['goal_a']} → {test_case['tactic_a']}")
        print(f"질문: {test_case['goal_b']} → ?")
        print(f"예상 답: {test_case['expected']}")
        
        try:
            # 여러 방법으로 아날로지 해결
            results = trainer.compare_analogy_methods(
                test_case['goal_a'], 
                test_case['tactic_a'], 
                test_case['goal_b'],
                top_k=3
            )
            
            for method_name, predictions in results.items():
                print(f"\n{method_name} 결과:")
                for j, (tactic, score) in enumerate(predictions, 1):
                    status = "✓" if tactic == test_case['expected'] else " "
                    print(f"  {status} {j}. {tactic}: {score:.3f}")
                    
        except Exception as e:
            print(f"   ✗ 에러 발생: {e}")
    
    print("\n4. 수학적 아날로지 예제들...")
    try:
        examples = trainer.mathematical_analogy_examples()
        for category, results in examples.items():
            print(f"\n{category}:")
            for tactic, score in results:
                print(f"  - {tactic}: {score:.3f}")
    except Exception as e:
        print(f"   ✗ 예제 실행 중 에러: {e}")
    
    print("\n=== 아날로지 테스트 완료 ===")

if __name__ == "__main__":
    test_analogy_methods()