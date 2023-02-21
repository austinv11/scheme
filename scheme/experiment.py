import jax
import jax.numpy as jnp

from scheme.util import consumes_key, jax_jit, generalized_logistic, StatefulPRNGKey


@consumes_key(3, 'key', count=2)
@jax_jit()
def simulate_batch_effect(expression_matrix: jnp.array,
                          p_dropout: float,
                          pcr_bias_stength: float,
                          key: StatefulPRNGKey):
    """
    Simulate batch effects by applying PCR bias, technical noise, and zero inflation.
    :param expression_matrix: The counts matrix.
    :param p_dropout: The random probability of dropout.
    :param pcr_bias_stength: The strength of the PCR bias.
    :param key: The random key.
    :return: The noisy counts matrix.
    """
    # PCR bias by making random low expression genes less likely to be detected and high expression genes more likely to be detected
    mean_expression = jnp.mean(expression_matrix)
    # Apply logistic function to simulate amplification bias
    # Center it on the mean expression, and scale it by 2 so that the range is 0-2
    # Finally, scale the growth rate by pcr_bias_strength (higher values mean more bias)
    # TODO: Test values from 0-1 to get a default, currently 0.25 is the default
    pcr_bias_mask = generalized_logistic(expression_matrix, 2, pcr_bias_stength, mean_expression)
    # Apply the bias
    expression_matrix = expression_matrix * pcr_bias_mask

    # Apply gaussian noise to simulate technical artifacts
    expression_matrix += expression_matrix * jax.random.normal(key[1], expression_matrix.shape)

    # Zero inflation
    zero_mask = jax.random.bernoulli(key[0], p_dropout, expression_matrix.shape)
    expression_matrix = expression_matrix * zero_mask

    # Clip the expression to make sure all values are >=0 and are integers
    expression_matrix = jnp.clip(expression_matrix, 0, jnp.inf)
    expression_matrix = jnp.round(expression_matrix)

    return expression_matrix