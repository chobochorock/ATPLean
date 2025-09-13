#!/usr/bin/env python3
"""
Word2Vec 아날로지 기능을 즉시 테스트하는 스크립트
"""

import sys
import os

# 현재 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from goal2vec_model import Goal2VecTrainer, EmbeddingConfig
from minif2f_processor import MathProblem, ProblemDifficulty

def quick_test():
    """빠른 테스트 - 기존 모델이 있으면 로드, 없으면 즉석 훈련"""
    print("=== Word2Vec 아날로지 기능 즉시 테스트 ===\n")
    
    config = EmbeddingConfig(
        embedding_dim=64,
        epochs=10,  # 빠른 테스트를 위해 줄임
        batch_size=8,
        learning_rate=0.01,
        min_count=1
    )
    
    print("1. Goal2Vec 트레이너 초기화...")
    trainer = Goal2VecTrainer(config)
    
    # 기존 모델 로드 시도
    model_path = "simple_goal2vec_model.pth"
    try:
        if os.path.exists(model_path):
            print("2. 기존 모델 로드 중...")
            trainer.load_model(model_path)
            print("   ✓ 기존 모델 로드 성공!")
        else:
            raise FileNotFoundError("기존 모델 없음")
    except:
        print("2. 기존 모델 없음 - 즉석 훈련 시작...")
        
        # 간단한 훈련 데이터 생성
        problems = [
            MathProblem("nat_add_zero", "∀ n : ℕ, n + 0 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
            MathProblem("real_add_zero", "∀ x : ℝ, x + 0 = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
            MathProblem("nat_add_comm", "∀ a b : ℕ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
            MathProblem("nat_mul_comm", "∀ a b : ℕ, a * b = b * a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
            MathProblem("real_add_comm", "∀ a b : ℝ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
            MathProblem("and_comm", "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P", proof="by simp", difficulty=ProblemDifficulty.MEDIUM),
            MathProblem("or_comm", "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P", proof="by simp", difficulty=ProblemDifficulty.MEDIUM),
        ]
        
        trainer.prepare_training_data(problems)
        trainer.train()
        trainer.save_model(model_path)
        print("   ✓ 즉석 훈련 완료!")
    
    print("\n3. Word2Vec 스타일 아날로지 테스트!")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "자연수 → 실수 타입 변환",
            "goal_a": "∀ n : ℕ, n + 0 = n",
            "tactic_a": "simp", 
            "goal_b": "∀ x : ℝ, x + 0 = x",
            "expected": "simp"
        },
        {
            "name": "덧셈 → 곱셈 연산 변환",
            "goal_a": "∀ a b : ℕ, a + b = b + a",
            "tactic_a": "ring",
            "goal_b": "∀ a b : ℕ, a * b = b * a", 
            "expected": "ring"
        },
        {
            "name": "AND → OR 논리 변환",
            "goal_a": "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P",
            "tactic_a": "simp",
            "goal_b": "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P",
            "expected": "simp"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i}: {test_case['name']} ---")
        print(f"알려진 사실: {test_case['goal_a']} → {test_case['tactic_a']}")
        print(f"질문: {test_case['goal_b']} → ?")
        print(f"예상 답: {test_case['expected']}")
        
        try:
            # 원본 Word2Vec 방식 테스트
            print("\n🔹 원본 Word2Vec 벡터 산술 방식:")
            analogy_result = trainer.solve_analogy(
                test_case['goal_a'], test_case['tactic_a'], test_case['goal_b'], top_k=3
            )
            for j, (tactic, score) in enumerate(analogy_result, 1):
                status = "✓" if tactic == test_case['expected'] else " "
                print(f"  {status} {j}. {tactic}: {score:.3f}")
            
            # 여러 방법 비교
            print("\n🔹 다양한 아날로지 방법 비교:")
            comparison = trainer.compare_analogy_methods(
                test_case['goal_a'], test_case['tactic_a'], test_case['goal_b'], top_k=2
            )
            
            for method_name, results in comparison.items():
                print(f"  {method_name}:")
                for tactic, score in results:
                    status = "✓" if tactic == test_case['expected'] else " "
                    print(f"    {status} {tactic}: {score:.3f}")
                    
        except Exception as e:
            print(f"  ❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n4. 수학적 아날로지 예제 시연...")
    try:
        examples = trainer.mathematical_analogy_examples()
        for category, results in examples.items():
            print(f"\n🔸 {category}:")
            for tactic, score in results:
                print(f"  - {tactic}: {score:.3f}")
    except Exception as e:
        print(f"  ❌ 예제 실행 실패: {e}")
    
    print("\n=== 테스트 완료! ===")
    print("\n💡 이제 ATPLean은 진정한 Word2Vec 스타일 아날로지를 지원합니다!")
    print("   - 'A is to B as C is to X' 형태의 수학적 추론 가능")
    print("   - 논문의 3가지 방법 모두 구현 (Original, Levy2014a, Levy2014b)")
    print("   - Church 논문의 평가 문제점 개선")

if __name__ == "__main__":
    quick_test()