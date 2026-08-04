from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PromptGroup:
    anatomical: tuple[str, ...]
    morphological: tuple[str, ...]
    diagnostic: tuple[str, ...]

    def levels(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        return self.anatomical, self.morphological, self.diagnostic


def _anatomical(label: str, sites: tuple[str, ...]) -> tuple[str, ...]:
    return (
        f"a clinical photograph of the oral mucosa showing {label}",
        f"an intraoral image of {sites[0]} demonstrating {label}",
        f"a close-up photograph of {sites[1]} revealing {label}",
        f"an oral medicine examination image showing {label} at {sites[2]}",
        f"a clinical dental image centered on {sites[3]} with {label}",
        f"an unretouched intraoral photograph depicting {label} on {sites[4]}",
    )


PROMPTS: Mapping[str, PromptGroup] = {
    "herpes_simplex": PromptGroup(
        _anatomical("herpes simplex", ("attached gingiva", "hard palate", "keratinised mucosa", "vermilion border", "dorsal tongue")),
        (
            "clustered vesicles on an erythematous base affecting keratinised mucosa",
            "small ulcers with scalloped borders in a herpetiform distribution pattern",
            "coalescing shallow erosions with irregular margins on the hard palate",
            "grouped translucent vesicles that rupture into shallow painful ulcerations",
            "recurrent crusted vesicles at the vermilion border with surrounding erythema",
            "multiple pinhead erosions merging into polycyclic ulcers on attached mucosa",
        ),
        (
            "oral lesion consistent with herpes simplex virus characterised by vesicular eruption on hard palate and attached gingiva, distinct from aphthous ulceration which favours non-keratinised mucosa",
            "recurrent intraoral herpes with clustered lesions on keratinised tissue rather than a solitary aphthous ulcer on mobile mucosa",
            "herpetic gingivostomatitis with diffuse painful gingivitis and vesicles rather than plaque-induced marginal gingivitis",
            "herpes labialis with grouped recurrent vesicles and crusting rather than an isolated traumatic ulcer",
            "HSV-pattern ulceration with scalloped coalescent borders rather than a round ulcer with a discrete erythematous halo",
            "vesicular infectious lesion favouring attached gingiva and palate rather than immune-mediated ulceration of buccal or labial mucosa",
        ),
    ),
    "aphthous_ulcer": PromptGroup(
        _anatomical("aphthous ulceration", ("buccal mucosa", "ventral tongue", "labial mucosa", "floor of mouth", "soft palate")),
        (
            "discrete round painful ulcer with a grey fibrinous base and erythematous halo",
            "shallow oval ulcer on non-keratinised mobile oral mucosa",
            "multiple tiny herpetiform aphthae without preceding intact vesicles",
            "well-demarcated recurrent ulcer with yellow-white pseudomembrane",
            "minor aphthous lesion under one centimetre with symmetric red margin",
            "major aphthous ulcer with deep crater and prolonged mucosal inflammation",
        ),
        (
            "aphthous ulceration on non-keratinised buccal mucosa distinct from recurrent herpes on attached gingiva or hard palate",
            "round discrete fibrin-covered ulcer without clustered preceding vesicles, favouring aphthous disease",
            "herpetiform aphthous ulcers distributed on mobile mucosa despite visual similarity to HSV lesions",
            "recurrent aphthous stomatitis without diffuse gingivitis or systemic primary herpetic eruption",
            "immune-mediated oral ulcer with a regular halo rather than irregular coalescing herpetic erosions",
            "aphthous lesion separated from traumatic ulcer by recurrence pattern and absence of a local mechanical cause",
        ),
    ),
    "periodontitis": PromptGroup(
        _anatomical("periodontitis", ("attached gingiva", "interdental papillae", "alveolar crest", "marginal gingiva", "periodontal tissues")),
        (
            "gingival inflammation with loss of stippling and blunted interdental papillae",
            "clinical attachment loss with visible recession of the marginal gingiva",
            "deep periodontal pockets with inflamed oedematous gingival margins",
            "alveolar bone loss accompanying chronic marginal inflammation",
            "tooth migration and exposed root surfaces caused by attachment loss",
            "bleeding periodontal tissues with recession and loss of papillary architecture",
        ),
        (
            "clinical attachment loss with probing depths exceeding three millimetres and radiographic alveolar bone loss, distinct from gingivitis without attachment loss",
            "periodontitis with destructive supporting-tissue loss rather than reversible marginal inflammation",
            "chronic periodontal disease with recession and bone loss rather than isolated gingival erythema",
            "inflamed gingiva plus loss of periodontal attachment separating disease from plaque-associated gingivitis",
            "destructive periodontal lesion affecting alveolar support rather than a superficial mucosal lesion",
            "periodontal pocketing and attachment loss consistent with periodontitis rather than healthy intact sulci",
        ),
    ),
    "gingivitis": PromptGroup(
        _anatomical("gingivitis", ("marginal gingiva", "interdental papillae", "free gingival margin", "attached gingiva", "gingival sulcus")),
        (
            "marginal gingival redness without attachment loss or bone resorption",
            "erythematous and oedematous marginal gingiva with spontaneous bleeding",
            "rounded shiny interdental papillae with loss of normal stippling",
            "plaque-associated swelling confined to the free gingival margin",
            "diffuse red gingiva that bleeds on gentle periodontal probing",
            "soft enlarged gingival tissue with altered contour but no recession",
        ),
        (
            "marginal inflammation without clinical attachment loss, distinct from destructive periodontitis",
            "reversible plaque-induced gingivitis with redness and bleeding but preserved alveolar support",
            "early gingival inflammation distinguished from healthy firm coral-pink stippled tissue",
            "generalised marginal gingivitis rather than focal ulcerative or vesicular mucosal disease",
            "gingival oedema and bleeding without pocketing caused by attachment loss",
            "surface gingival inflammatory change without radiographic alveolar bone destruction",
        ),
    ),
    "healthy_mucosa": PromptGroup(
        _anatomical("healthy oral mucosa", ("attached gingiva", "hard palate", "buccal mucosa", "ventral tongue", "oral cavity")),
        (
            "normal coral-pink gingiva with intact stippling and knife-edge interdental papillae",
            "uniform moist oral mucosa without ulceration plaque swelling or pigment change",
            "firm gingiva with regular scalloped margins and no bleeding",
            "intact epithelium with physiologic colour and symmetric surface texture",
            "healthy oral tissues without vesicles erosions attachment loss or oedema",
            "normal mucosal folds and vascular pattern without focal pathological lesion",
        ),
        (
            "intact gingival architecture with uniform coral-pink colour and firm consistency, distinct from early gingivitis with subtle colour change and mild marginal oedema",
            "normal mucosa without focal ulceration distinguished from healed or subclinical inflammatory lesions",
            "healthy gingiva with stippling and sharp papillae rather than swollen rounded bleeding margins",
            "uninterrupted oral epithelium without vesicular infectious or aphthous disease",
            "physiological pigmentation without irregular white red or ulcerative change",
            "clinically healthy periodontal tissue without attachment loss recession or inflammation",
        ),
    ),
    "leukoplakia": PromptGroup(
        _anatomical("oral leukoplakia", ("buccal mucosa", "lateral tongue", "floor of mouth", "gingiva", "oral mucosa")),
        (
            "persistent adherent white plaque with sharply demarcated irregular surface",
            "homogeneous opaque keratotic patch that cannot be wiped away",
            "non-homogeneous speckled white and red mucosal plaque",
            "corrugated thickened keratinised lesion with fissured surface",
            "well-defined white mucosal patch without surrounding vesicles",
            "verrucous leukokeratotic plaque with uneven exophytic texture",
        ),
        (
            "persistent non-scrapable white plaque consistent with leukoplakia rather than removable candidal pseudomembrane",
            "keratotic mucosal lesion requiring exclusion of frictional trauma and lichen planus",
            "oral potentially malignant white lesion rather than normal physiologic keratinisation",
            "homogeneous leukoplakia distinguished from ulcerative inflammatory and vesicular disorders",
            "speckled erythroleukoplakia with mixed red-white change rather than isolated erythema",
            "persistent focal plaque warranting specialist assessment rather than transient surface debris",
        ),
    ),
    "caries": PromptGroup(
        _anatomical("dental caries", ("occlusal tooth surface", "proximal enamel", "cervical tooth surface", "dentin", "posterior dentition")),
        (
            "dark cavitated enamel lesion with exposed softened dentin",
            "chalky white enamel demineralisation adjacent to a plaque stagnation area",
            "brown fissure lesion with breakdown of the occlusal surface",
            "proximal carious shadow beneath an undermined marginal ridge",
            "cervical cavitation with irregular discoloured tooth structure",
            "advanced coronal decay with loss of enamel and dentinal involvement",
        ),
        (
            "demineralised or cavitated tooth structure consistent with caries rather than external calculus",
            "intrinsic enamel-dentin breakdown distinguished from removable plaque deposits",
            "carious lesion localised within tooth structure rather than marginal gingival inflammation",
            "occlusal cavitation with softened dentin rather than a stained intact fissure",
            "proximal caries suggested by enamel shadow and marginal ridge change",
            "active decay with matte demineralised surface rather than arrested hard glossy lesion",
        ),
    ),
    "calculus": PromptGroup(
        _anatomical("dental calculus", ("lingual lower incisors", "buccal upper molars", "gingival margin", "tooth root", "interproximal tooth surface")),
        (
            "hard yellow-brown mineralised deposit adherent to the tooth surface",
            "supragingival calculus ledge adjacent to an inflamed gingival margin",
            "dark subgingival mineral deposit visible along an exposed root",
            "irregular calcified plaque mass bridging interproximal surfaces",
            "chalky adherent accretion concentrated near a salivary duct opening",
            "rough layered tooth-surface deposit with marginal gingival irritation",
        ),
        (
            "adherent mineralised deposit external to enamel distinguished from cavitated dental caries",
            "calculus accumulation at gingival margin rather than intrinsic tooth discoloration",
            "hard calcified plaque associated with local gingivitis but separate from soft tissue disease",
            "supragingival deposit visible above the margin rather than subgingival root caries",
            "rough mineral accretion that cannot be removed by brushing unlike soft plaque",
            "tooth-surface calculus with secondary gingival inflammation rather than primary mucosal lesion",
        ),
    ),
}


def validate_prompt_library(library: Mapping[str, PromptGroup] = PROMPTS) -> None:
    if not library:
        raise ValueError("prompt library is empty")
    for label, group in library.items():
        if not label:
            raise ValueError("prompt label is empty")
        for level in group.levels():
            if len(level) != 6:
                raise ValueError(f"{label} requires six prompts per level")
            if any(not prompt.strip() for prompt in level):
                raise ValueError(f"{label} contains an empty prompt")
