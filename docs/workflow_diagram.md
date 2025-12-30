# Workflow Diagram with New Folder Structure

## Script Workflow with Output Organization

```mermaid
graph TD
    A[User runs script] --> B{Which script?}
    
    B -->|compute_wrmd.py| C[Create outputs/compute_wrmd/]
    B -->|extract_activations.py| D[Create outputs/extract_activations/]
    B -->|find_best_layers.py| E[Create outputs/find_best_layers/]
    
    C --> C1[Process activations]
    C1 --> C2[Generate steering vectors]
    C2 --> C3[Save to outputs/compute_wrmd/]
    C3 --> C4[steering_vectors_*.pt]
    C3 --> C5[steering_vector_norms.png]
    C3 --> C6[method_comparison.png]
    
    D --> D1[Extract activations from model]
    D1 --> D2[Process prompts and labels]
    D2 --> D3[Save to outputs/extract_activations/]
    D3 --> D4[activations_*.pt]
    
    E --> E1[Load activations and steering vectors]
    E1 --> E2[Compute correlations]
    E2 --> E3[Generate plots]
    E3 --> E4[Save to outputs/find_best_layers/]
    E4 --> E5[layer_correlations.json]
    E4 --> E6[layer_correlations.png]
    E4 --> E7[layer_*_projection_scatter.png]
    
    C4 --> F[Clean project root]
    C5 --> F
    C6 --> F
    D4 --> F
    E5 --> F
    E6 --> F
    E7 --> F
```

## Data Flow Between Scripts

```mermaid
graph LR
    subgraph extract_activations
        A1[extract_activations.py] --> A2[outputs/extract_activations/activations_*.pt]
    end
    
    subgraph compute_wrmd
        B1[compute_wrmd.py] --> B2[outputs/compute_wrmd/steering_vectors_*.pt]
    end
    
    subgraph find_best_layers
        C1[find_best_layers.py] --> C2[outputs/find_best_layers/analysis_results]
    end
    
    A2 --> B1
    A2 --> C1
    B2 --> C1
    
    style A1 fill:#e1f5fe
    style B1 fill:#e8f5e9
    style C1 fill:#fff3e0
    style A2 fill:#e1f5fe
    style B2 fill:#e8f5e9
    style C2 fill:#fff3e0