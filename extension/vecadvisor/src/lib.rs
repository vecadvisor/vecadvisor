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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_document_is_metadata_only() {
        let document = capability_document();

        assert_eq!(document["extension"], EXTENSION_NAME);
        assert_eq!(document["extension_version"], EXTENSION_VERSION);
        assert_eq!(document["planner_changes_enabled"], false);
        assert_eq!(
            document["capabilities"]
                .as_array()
                .expect("capabilities should be an array")
                .len(),
            4
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
}
