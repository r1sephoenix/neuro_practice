"""
Command-line interface for BCI classification
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from .classifier import MultiClassifierComparison
from .pipeline import BCIPipeline


@click.group()
@click.version_option(version="0.1.0")
def main():
    """BCI Classifier - Brain-Computer Interface classification using MNE Python."""
    pass


@main.command()
@click.option('--dataset', '-d', 
              type=click.Choice(['eegbci', 'sample']), 
              default='eegbci',
              help='Dataset to use for classification')
@click.option('--subject', '-s', 
              type=int, 
              default=1,
              help='Subject number (for datasets that support multiple subjects)')
@click.option('--feature-method', '-f',
              type=click.Choice(['csp', 'psd', 'time_domain', 'combined']),
              default='csp',
              help='Feature extraction method')
@click.option('--classifier', '-c',
              type=click.Choice(['svm', 'rf', 'lda', 'lr', 'knn']),
              default='svm',
              help='Classification algorithm')
@click.option('--l-freq', 
              type=float, 
              default=8.0,
              help='Low frequency for band-pass filter (Hz)')
@click.option('--h-freq', 
              type=float, 
              default=30.0,
              help='High frequency for band-pass filter (Hz)')
@click.option('--tmin', 
              type=float, 
              default=-1.0,
              help='Start time before event (seconds)')
@click.option('--tmax', 
              type=float, 
              default=4.0,
              help='End time after event (seconds)')
@click.option('--test-size', 
              type=float, 
              default=0.2,
              help='Proportion of data for testing (0-1)')
@click.option('--no-ica', 
              is_flag=True,
              help='Skip ICA artifact removal')
@click.option('--no-normalize', 
              is_flag=True,
              help='Skip feature normalization')
@click.option('--output', '-o',
              type=click.Path(),
              help='Path to save the trained pipeline')
@click.option('--random-state', 
              type=int, 
              default=42,
              help='Random state for reproducibility')
@click.option('--runs',
              type=str,
              default="6,10,14",
              help='Comma-separated list of runs for EEG BCI dataset (e.g., "6,10,14")')
def classify(dataset, subject, feature_method, classifier, l_freq, h_freq, 
            tmin, tmax, test_size, no_ica, no_normalize, output, random_state, runs):
    """Run BCI classification pipeline."""
    
    click.echo("🧠 BCI Classifier - Starting classification pipeline")
    click.echo("=" * 60)
    
    # Parse runs parameter
    if dataset == 'eegbci':
        try:
            runs_list = [int(r.strip()) for r in runs.split(',')]
        except ValueError:
            click.echo("❌ Error: Invalid runs format. Use comma-separated integers (e.g., '6,10,14')")
            sys.exit(1)
    else:
        runs_list = None
    
    # Display configuration
    click.echo(f"📊 Dataset: {dataset}")
    click.echo(f"👤 Subject: {subject}")
    if runs_list:
        click.echo(f"🏃 Runs: {runs_list}")
    click.echo(f"🔧 Feature method: {feature_method}")
    click.echo(f"🤖 Classifier: {classifier}")
    click.echo(f"🎛️  Filter: {l_freq}-{h_freq} Hz")
    click.echo(f"⏱️  Epoch: {tmin} to {tmax} seconds")
    click.echo(f"📋 Test size: {test_size}")
    click.echo(f"🔄 Apply ICA: {not no_ica}")
    click.echo(f"📐 Normalize: {not no_normalize}")
    click.echo()
    
    try:
        # Initialize pipeline
        pipeline = BCIPipeline(
            dataset=dataset,
            feature_method=feature_method,
            classifier_algorithm=classifier,
            random_state=random_state
        )
        
        # Prepare kwargs for data loading
        kwargs = {}
        if runs_list:
            kwargs['runs'] = runs_list
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            subject=subject,
            tmin=tmin,
            tmax=tmax,
            l_freq=l_freq,
            h_freq=h_freq,
            test_size=test_size,
            apply_ica=not no_ica,
            normalize_features=not no_normalize,
            **kwargs
        )
        
        # Display results
        click.echo()
        click.echo("📊 CLASSIFICATION RESULTS")
        click.echo("=" * 30)
        click.echo(f"🎯 Test Accuracy: {results['test_accuracy']:.3f}")
        click.echo(f"📊 CV Accuracy: {results['cv_mean_accuracy']:.3f} ± {results['cv_std_accuracy']:.3f}")
        
        # Save pipeline if requested
        if output:
            output_path = Path(output)
            pipeline.save_pipeline(str(output_path))
            click.echo(f"💾 Pipeline saved to: {output_path}")
        
        # Display summary
        summary = pipeline.get_pipeline_summary()
        click.echo()
        click.echo("📋 PIPELINE SUMMARY")
        click.echo("=" * 20)
        if summary.get('feature_info'):
            feature_info = summary['feature_info']
            click.echo(f"Features: {feature_info['n_features']}")
            click.echo(f"Training samples: {feature_info['n_train_samples']}")
            click.echo(f"Test samples: {feature_info['n_test_samples']}")
        
        click.echo()
        click.echo("Classification completed successfully")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--dataset', '-d', 
              type=click.Choice(['eegbci', 'sample']), 
              default='eegbci',
              help='Dataset to use for classification')
@click.option('--subject', '-s', 
              type=int, 
              default=1,
              help='Subject number')
@click.option('--feature-method', '-f',
              type=click.Choice(['csp', 'psd', 'time_domain', 'combined']),
              default='csp',
              help='Feature extraction method')
@click.option('--algorithms',
              type=str,
              default="svm,rf,lda,lr,knn",
              help='Comma-separated list of algorithms to compare')
@click.option('--cv-folds', 
              type=int, 
              default=5,
              help='Number of cross-validation folds')
@click.option('--l-freq', 
              type=float, 
              default=8.0,
              help='Low frequency for band-pass filter (Hz)')
@click.option('--h-freq', 
              type=float, 
              default=30.0,
              help='High frequency for band-pass filter (Hz)')
@click.option('--no-ica', 
              is_flag=True,
              help='Skip ICA artifact removal')
@click.option('--random-state', 
              type=int, 
              default=42,
              help='Random state for reproducibility')
@click.option('--runs',
              type=str,
              default="6,10,14",
              help='Comma-separated list of runs for EEG BCI dataset')
def compare(dataset, subject, feature_method, algorithms, cv_folds, 
           l_freq, h_freq, no_ica, random_state, runs):
    """Compare different classification algorithms."""
    
    click.echo("🤖 BCI Classifier - Algorithm Comparison")
    click.echo("=" * 50)
    
    # Parse algorithms and runs
    try:
        algorithms_list = [alg.strip() for alg in algorithms.split(',')]
        runs_list = [int(r.strip()) for r in runs.split(',')] if dataset == 'eegbci' else None
    except ValueError:
        click.echo("❌ Error: Invalid format. Check algorithms and runs parameters.")
        sys.exit(1)
    
    click.echo(f"Dataset: {dataset} (subject {subject})")
    if runs_list:
        click.echo(f" Runs: {runs_list}")
    click.echo(f" Feature method: {feature_method}")
    click.echo(f" Algorithms: {', '.join(algorithms_list)}")
    click.echo(f" CV folds: {cv_folds}")
    click.echo()
    
    try:
        # Initialize pipeline with first algorithm (will be changed during comparison)
        pipeline = BCIPipeline(
            dataset=dataset,
            feature_method=feature_method,
            classifier_algorithm=algorithms_list[0],
            random_state=random_state
        )
        
        # Prepare data
        kwargs = {}
        if runs_list:
            kwargs['runs'] = runs_list
            
        pipeline.load_data(subject=subject, **kwargs)
        pipeline.create_epochs()
        pipeline.preprocess_data(l_freq=l_freq, h_freq=h_freq, apply_ica=not no_ica)
        pipeline.split_data()
        pipeline.extract_features()
        
        # Compare algorithms
        results = pipeline.compare_algorithms(algorithms=algorithms_list, cv=cv_folds)
        
        # Display results
        click.echo()
        click.echo("COMPARISON RESULTS")
        click.echo("=" * 25)
        
        # Sort algorithms by performance
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['mean_accuracy'], 
                               reverse=True)
        
        for i, (alg, result) in enumerate(sorted_results):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            click.echo(f"{emoji} {alg.upper()}: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
        
        best_alg = sorted_results[0][0]
        best_score = sorted_results[0][1]['mean_accuracy']
        click.echo()
        click.echo(f" Best algorithm: {best_alg.upper()} ({best_score:.3f})")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


@main.command()
@click.argument('pipeline_path', type=click.Path(exists=True))
@click.option('--test-data',
              type=click.Path(exists=True),
              help='Path to test data (if different from training)')
def evaluate(pipeline_path, test_data):
    """Evaluate a saved pipeline."""
    
    click.echo(" BCI Classifier - Pipeline Evaluation")
    click.echo("=" * 45)
    
    try:
        # Load pipeline
        pipeline = BCIPipeline()
        pipeline.load_pipeline(pipeline_path)
        
        click.echo(f"Loaded pipeline from: {pipeline_path}")
        
        # Get summary
        summary = pipeline.get_pipeline_summary()
        
        click.echo()
        click.echo(" PIPELINE CONFIGURATION")
        click.echo("=" * 30)
        config = summary['configuration']
        click.echo(f" Dataset: {config['dataset']}")
        click.echo(f" Feature method: {config['feature_method']}")
        click.echo(f" Classifier: {config['classifier_algorithm']}")
        
        if summary.get('results'):
            results = summary['results']
            click.echo()
            click.echo(" SAVED RESULTS")
            click.echo("=" * 20)
            click.echo(f" Test Accuracy: {results.get('test_accuracy', 'N/A')}")
            click.echo(f" CV Accuracy: {results.get('cv_mean_accuracy', 'N/A')}")
            
            if 'cv_std_accuracy' in results:
                click.echo(f" CV Std: ±{results['cv_std_accuracy']:.3f}")
        
        if summary.get('feature_info'):
            feature_info = summary['feature_info']
            click.echo()
            click.echo(" FEATURE INFORMATION")
            click.echo("=" * 25)
            click.echo(f"Features: {feature_info['n_features']}")
            click.echo(f"Training samples: {feature_info['n_train_samples']}")
            if feature_info.get('n_test_samples'):
                click.echo(f"Test samples: {feature_info['n_test_samples']}")
        
    except Exception as e:
        click.echo(f" Error: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--dataset', '-d', 
              type=click.Choice(['eegbci', 'sample']), 
              default='eegbci',
              help='Dataset to use')
@click.option('--subject', '-s', 
              type=int, 
              default=1,
              help='Subject number')
@click.option('--runs',
              type=str,
              default="6,10,14",
              help='Comma-separated list of runs for EEG BCI dataset')
def info(dataset, subject, runs):
    """Get information about a dataset."""
    
    click.echo("BCI Classifier - Dataset Information")
    click.echo("=" * 45)
    
    try:
        from .data_loader import DataLoader

        # Parse runs
        runs_list = None
        if dataset == 'eegbci':
            try:
                runs_list = [int(r.strip()) for r in runs.split(',')]
            except ValueError:
                click.echo("❌ Error: Invalid runs format")
                sys.exit(1)
        
        # Load data
        loader = DataLoader()
        
        if dataset == 'eegbci':
            raw, events, event_id = loader.load_eegbci_data(subject=subject, runs=runs_list)
        else:
            raw, events, event_id = loader.load_sample_data()
        
        # Get info
        info = loader.get_info()
        
        click.echo(f"Dataset: {dataset}")
        click.echo(f"Subject: {subject}")
        if runs_list:
            click.echo(f"🏃 Runs: {runs_list}")
        click.echo()
        click.echo("DATA INFORMATION")
        click.echo("=" * 20)
        click.echo(f"Channels: {info['n_channels']}")
        click.echo(f"Sampling rate: {info['sampling_rate']} Hz")
        click.echo(f"Duration: {info['duration']:.1f} seconds")
        click.echo(f"Events: {info['n_events']}")
        click.echo(f"Event types: {', '.join(info['event_types'])}")
        
        click.echo()
        click.echo(" CHANNEL NAMES")
        click.echo("=" * 16)
        channel_names = info['channel_names']
        # Display channels in rows of 8
        for i in range(0, len(channel_names), 8):
            row = channel_names[i:i+8]
            click.echo(f"{', '.join(row)}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()