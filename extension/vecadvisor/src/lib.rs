#![cfg_attr(not(feature = "pgrx"), allow(dead_code))]

#[cfg(feature = "pgrx")]
use pgrx::JsonB;
#[cfg(feature = "pgrx")]
use pgrx::prelude::*;
use serde_json::{Value, json};

#[cfg(feature = "pgrx")]
pgrx::pg_module_magic!();

const EXTENSION_NAME: &str = "vecadvisor";
const EXTENSION_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PACKAGE_NAME: &str = "vecadvisor";
const DEFAULT_RECALL_AT_EF: f64 = 1.0;

#[derive(Debug, Clone, Eq, PartialEq)]
struct Capability {
    name: &'static str,
    enabled: bool,
    detail: &'static str,
}

fn capabilities() -> Vec<Capability> {
    vec![
        Capability {
            name: "sql_metadata_functions",
            enabled: true,
            detail: "vecadvisor_extension_version() and vecadvisor_capabilities() are available",
        },
        Capability {
            name: "postfilter_risk_estimator",
            enabled: true,
            detail: "vecadvisor_postfilter_risk() estimates candidate survival from selectivity inputs",
        },
        Capability {
            name: "spi_catalog_probes",
            enabled: false,
            detail: "planned; will use read-only SPI/catalog access with statement timeouts",
        },
        Capability {
            name: "planner_hooks",
            enabled: false,
            detail: "not installed in the scaffold; future work must be opt-in and guarded by GUCs",
        },
        Capability {
            name: "python_cli_parity",
            enabled: false,
            detail: "planned; extension recommendations must match Python CLI fixtures before use",
        },
    ]
}

fn capability_document() -> Value {
    let capability_values: Vec<Value> = capabilities()
        .into_iter()
        .map(|capability| {
            json!({
                "name": capability.name,
                "enabled": capability.enabled,
                "detail": capability.detail,
            })
        })
        .collect();

    json!({
        "extension": EXTENSION_NAME,
        "extension_version": EXTENSION_VERSION,
        "python_package": PYTHON_PACKAGE_NAME,
        "planner_changes_enabled": false,
        "capabilities": capability_values,
    })
}

#[cfg(feature = "pgrx")]
#[pg_extern]
fn vecadvisor_extension_version() -> &'static str {
    EXTENSION_VERSION
}

#[cfg(feature = "pgrx")]
#[pg_extern]
fn vecadvisor_capabilities() -> JsonB {
    JsonB(capability_document())
}

#[cfg(feature = "pgrx")]
#[pg_extern]
fn vecadvisor_postfilter_risk(
    limit_count: i32,
    ef_search: i32,
    global_selectivity: f64,
    local_selectivity: Option<f64>,
    recall_at_ef: Option<f64>,
) -> JsonB {
    match postfilter_risk_document(
        limit_count,
        ef_search,
        global_selectivity,
        local_selectivity,
        recall_at_ef,
    ) {
        Ok(document) => JsonB(document),
        Err(message) => pgrx::error!("{}", message),
    }
}

fn postfilter_risk_document(
    limit_count: i32,
    ef_search: i32,
    global_selectivity: f64,
    local_selectivity: Option<f64>,
    recall_at_ef: Option<f64>,
) -> Result<Value, String> {
    validate_positive_i32("limit_count", limit_count)?;
    validate_positive_i32("ef_search", ef_search)?;
    validate_probability("global_selectivity", global_selectivity)?;
    if let Some(value) = local_selectivity {
        validate_probability("local_selectivity", value)?;
    }
    if let Some(value) = recall_at_ef {
        validate_probability("recall_at_ef", value)?;
    }

    let effective_selectivity = local_selectivity.unwrap_or(global_selectivity);
    let selectivity_source = if local_selectivity.is_some() {
        "local"
    } else {
        "global_fallback"
    };
    let recall_multiplier = recall_at_ef.unwrap_or(DEFAULT_RECALL_AT_EF);
    let expected_survivors = effective_selectivity * f64::from(ef_search);
    let survivor_ratio = expected_survivors / f64::from(limit_count);
    let returns_k = expected_survivors >= f64::from(limit_count);
    let estimated_recall = recall_multiplier * survivor_ratio.min(1.0);
    let risk_level = postfilter_risk_level(survivor_ratio, selectivity_source);
    let recommended_min_ef = required_survivor_ef(effective_selectivity, limit_count);

    let mut notes = vec![format!(
        "costing uses {} selectivity; expected survivors ~= {:.2}",
        selectivity_source, expected_survivors
    )];
    if local_selectivity.is_none() {
        notes.push(
            "local selectivity was not supplied; global selectivity can be wrong for correlated filters"
                .to_string(),
        );
    }
    if recall_at_ef.is_none() {
        notes.push("recall_at_ef was not supplied; survivor-bound recall assumes 1.0 ANN recall multiplier".to_string());
    }

    let recommendations =
        postfilter_recommendations(returns_k, recommended_min_ef, ef_search, selectivity_source);

    Ok(json!({
        "extension": EXTENSION_NAME,
        "kind": "postfilter_risk",
        "planner_changes_enabled": false,
        "inputs": {
            "limit": limit_count,
            "ef_search": ef_search,
            "global_selectivity": global_selectivity,
            "local_selectivity": local_selectivity,
            "recall_at_ef": recall_multiplier,
            "selectivity_source": selectivity_source,
        },
        "estimates": {
            "expected_survivors": expected_survivors,
            "survivor_ratio": survivor_ratio,
            "returns_k": returns_k,
            "estimated_recall": estimated_recall,
            "recommended_min_ef_for_survivors": recommended_min_ef,
        },
        "risk": {
            "level": risk_level,
            "reason": postfilter_risk_reason(risk_level, returns_k, selectivity_source),
        },
        "recommendations": recommendations,
        "notes": notes,
    }))
}

fn validate_positive_i32(name: &str, value: i32) -> Result<(), String> {
    if value <= 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(())
}

fn validate_probability(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{name} must be a finite value in [0, 1]"));
    }
    Ok(())
}

fn required_survivor_ef(selectivity: f64, limit_count: i32) -> i32 {
    if selectivity <= 0.0 {
        return i32::MAX;
    }
    (f64::from(limit_count) / selectivity)
        .ceil()
        .min(f64::from(i32::MAX)) as i32
}

fn postfilter_risk_level(survivor_ratio: f64, selectivity_source: &str) -> &'static str {
    if survivor_ratio < 1.0 {
        "high"
    } else if survivor_ratio < 2.0 || selectivity_source == "global_fallback" {
        "medium"
    } else {
        "low"
    }
}

fn postfilter_risk_reason(
    risk_level: &str,
    returns_k: bool,
    selectivity_source: &str,
) -> &'static str {
    if !returns_k {
        "expected post-filter survivors are below LIMIT"
    } else if selectivity_source == "global_fallback" {
        "candidate survival is based on global selectivity because no local probe was supplied"
    } else if risk_level == "medium" {
        "expected survivors meet LIMIT but leave limited margin"
    } else {
        "expected survivors have margin above LIMIT under the supplied local selectivity"
    }
}

fn postfilter_recommendations(
    returns_k: bool,
    recommended_min_ef: i32,
    ef_search: i32,
    selectivity_source: &str,
) -> Vec<String> {
    let mut recommendations = Vec::new();
    if !returns_k {
        recommendations.push(format!(
            "raise hnsw.ef_search to at least {recommended_min_ef} for expected survivors >= LIMIT"
        ));
        recommendations.push(
            "consider iterative_scan, filter-first exact search, a partial HNSW index, or partitioning"
                .to_string(),
        );
    } else if recommended_min_ef > ef_search {
        recommendations.push(format!(
            "raise hnsw.ef_search toward {recommended_min_ef} before relying on post-filter ANN"
        ));
    } else {
        recommendations.push(
            "post-filter ANN has enough expected survivors under the supplied selectivity"
                .to_string(),
        );
    }
    if selectivity_source == "global_fallback" {
        recommendations.push(
            "run a local-selectivity probe before treating this as a safe recommendation"
                .to_string(),
        );
    }
    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_document_reports_safe_surface() {
        let document = capability_document();

        assert_eq!(document["extension"], EXTENSION_NAME);
        assert_eq!(document["extension_version"], EXTENSION_VERSION);
        assert_eq!(document["planner_changes_enabled"], false);
        assert_eq!(
            document["capabilities"]
                .as_array()
                .expect("capabilities should be an array")
                .len(),
            5
        );
    }

    #[test]
    fn scaffold_keeps_planner_hooks_disabled() {
        let planner_hook = capabilities()
            .into_iter()
            .find(|capability| capability.name == "planner_hooks")
            .expect("planner hook capability should be reported");

        assert!(!planner_hook.enabled);
        assert!(planner_hook.detail.contains("not installed"));
    }

    #[test]
    fn postfilter_risk_uses_local_selectivity_for_survivors() {
        let document = postfilter_risk_document(10, 40, 0.20, Some(0.05), Some(0.9))
            .expect("risk document should be valid");

        assert_eq!(document["inputs"]["selectivity_source"], "local");
        assert_eq!(document["estimates"]["expected_survivors"], 2.0);
        assert_eq!(document["estimates"]["returns_k"], false);
        assert_eq!(
            document["estimates"]["recommended_min_ef_for_survivors"],
            200
        );
        assert_eq!(document["risk"]["level"], "high");
        assert_eq!(
            document["estimates"]["estimated_recall"],
            0.18000000000000002
        );
    }

    #[test]
    fn postfilter_risk_marks_global_fallback_as_medium_even_when_survivors_pass() {
        let document = postfilter_risk_document(10, 100, 0.20, None, None)
            .expect("risk document should be valid");

        assert_eq!(document["inputs"]["selectivity_source"], "global_fallback");
        assert_eq!(document["estimates"]["expected_survivors"], 20.0);
        assert_eq!(document["estimates"]["returns_k"], true);
        assert_eq!(document["risk"]["level"], "medium");
        assert!(
            document["recommendations"][1]
                .as_str()
                .expect("recommendation should be a string")
                .contains("local-selectivity probe")
        );
    }

    #[test]
    fn postfilter_risk_reports_low_when_local_survivors_have_margin() {
        let document = postfilter_risk_document(10, 200, 0.05, Some(0.20), Some(0.95))
            .expect("risk document should be valid");

        assert_eq!(document["estimates"]["expected_survivors"], 40.0);
        assert_eq!(document["estimates"]["returns_k"], true);
        assert_eq!(document["risk"]["level"], "low");
        assert_eq!(document["estimates"]["estimated_recall"], 0.95);
    }

    #[test]
    fn postfilter_risk_validates_inputs() {
        assert!(postfilter_risk_document(0, 40, 0.1, Some(0.1), None).is_err());
        assert!(postfilter_risk_document(10, 0, 0.1, Some(0.1), None).is_err());
        assert!(postfilter_risk_document(10, 40, -0.1, Some(0.1), None).is_err());
        assert!(postfilter_risk_document(10, 40, 0.1, Some(1.1), None).is_err());
        assert!(postfilter_risk_document(10, 40, 0.1, Some(0.1), Some(f64::NAN)).is_err());
    }
}
