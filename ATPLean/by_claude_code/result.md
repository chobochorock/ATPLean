Sending request to model: deepseek/deepseek-prover-v2...
------------------------------
--- Model Response ---
### Detailed Proof and Analysis

**Problem:** Find the positive difference between 120% of 30 and 130% of 20, and show that it is 10.

**Solution:**
1. Calculate 120% of 30:
   \[
   \frac{120}{100} \times 30 = 1.2 \times 30 = 36
   \]
2. Calculate 130% of 20:
   \[
   \frac{130}{100} \times 20 = 1.3 \times 20 = 26
   \]
3. Find the positive difference:
   \[
   36 - 26 = 10
   \]
4. The absolute value is not strictly necessary here since the difference is already positive, but the Lean statement uses `abs` to ensure the result is non-negative. In this case, `abs(36 - 26) = 10`.

However, the Lean statement is:
\[
\text{abs}\left(\frac{120}{100} \times 30 - \frac{130}{100} \times 20\right) = 10
\]
But notice that:
\[
\frac{120}{100} \times 30 - \frac{130}{100} \times 20 = 36 - 26 = 10
\]
So, the expression inside the absolute value is already positive, and the absolute value is redundant. But Lean's `abs` will still correctly return `10` because `abs(10) = 10`.

But wait! The Lean statement uses integer division for `120 / 100` and `130 / 100`, but the numbers are interpreted as reals (`(120 : ℝ)`). So:
\[
\frac{120}{100} = 1.2, \quad \frac{130}{100} = 1.3
\]
and the calculation is correct as written in Lean. The result is indeed `10`.

But let's double-check the calculation:
\[
\frac{120}{100} \times 30 - \frac{130}{100} \times 20 = 1.2 \times 30 - 1.3 \times 20 = 36 - 26 = 10
\]
The absolute value is unnecessary here because `10 ≥ 0`, but it doesn't hurt.

**Formal Proof Sketch:**
1. Compute `(120 : ℝ) / 100 * 30 = 36`.
2. Compute `(130 : ℝ) / 100 * 20 = 26`.
3. Compute the difference `36 - 26 = 10`.
4. Take the absolute value `abs(10) = 10`.

But Lean's `abs` is defined as a piecewise function:
- If `x ≥ 0`, then `abs x = x`.
- If `x < 0`, then `abs x = -x`.

Here, `(120 : ℝ) / 100 * 30 - 130 / 100 * 20 = 10 ≥ 0`, so `abs(10) = 10`.

**Step 1:** Compute `(120 : ℝ) / 100 * 30 - 130 / 100 * 20 = 10`.
**Step 2:** Since `10 ≥ 0`, `abs(10) = 10`.

But we can directly compute the expression inside the `abs` and check that it is `10`, and then the `abs` is trivial.

### Step-by-Step Abstract Plan

1. **Compute the expression inside the absolute value:**
   - Calculate `(120 : ℝ) / 100 * 30 = 36`.
   - Calculate `(130 : ℝ) / 100 * 20 = 26`.
   - Subtract to get `36 - 26 = 10`.

2. **Apply the absolute value:**
   - Since `10 ≥ 0`, `abs(10) = 10`.

### Lean 4 Proof with `have` Statements

```lean4
theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  have h₁ : (120 : ℝ) / 100 * 30 - 130 / 100 * 20 = 10 := by
    sorry
  have h₂ : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
    sorry
  sorry
```

### Complete Lean 4 Proof

```lean4
theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  have h₁ : (120 : ℝ) / 100 * 30 - 130 / 100 * 20 = 10 := by
    norm_num [div_eq_mul_inv, mul_assoc]
    <;> ring_nf at *
    <;> norm_num at *
    <;> linarith
  
  have h₂ : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
    rw [h₁]
    -- Since the expression inside the absolute value is 10, which is positive, the absolute value of 10 is 10.
    norm_num [abs_of_nonneg]
    <;> linarith
  
  exact h₂
```
------------------------------
Request completed in 9.22 seconds.
