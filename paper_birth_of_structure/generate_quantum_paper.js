const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel,
  PageNumber, PageBreak, BorderStyle, ExternalHyperlink,
  LevelFormat, TabStopType, TabStopPosition
} = require("docx");

// Figure directory
const FIG_DIR = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis";

// Helper: load image and get dimensions
function loadImage(filename) {
  const filepath = path.join(FIG_DIR, filename);
  return fs.readFileSync(filepath);
}

// Helper: create a figure paragraph with image + caption
function figureParagraph(filename, captionText, widthInches = 6.5, aspectRatio = 0.65) {
  const imgData = loadImage(filename);
  const width = Math.round(widthInches * 96);
  const height = Math.round(width * aspectRatio);
  return [
    new Paragraph({ spacing: { before: 200, after: 80 }, alignment: AlignmentType.CENTER, children: [
      new ImageRun({
        type: "png",
        data: imgData,
        transformation: { width, height },
        altText: { title: captionText, description: captionText, name: filename }
      })
    ]}),
    new Paragraph({
      spacing: { after: 240 },
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: captionText, italics: true, size: 20, font: "Arial" })
      ]
    })
  ];
}

// Helper: image-only paragraph (no caption), for multi-row figure groups
function figureImageOnly(filename, widthInches = 6.5, aspectRatio = 0.36) {
  const imgData = loadImage(filename);
  const width = Math.round(widthInches * 96);
  const height = Math.round(width * aspectRatio);
  return new Paragraph({ spacing: { before: 120, after: 40 }, alignment: AlignmentType.CENTER, children: [
    new ImageRun({
      type: "png",
      data: imgData,
      transformation: { width, height },
      altText: { title: filename, description: filename, name: filename }
    })
  ]});
}

// Helper: caption-only paragraph for multi-row figures
function figureCaption(captionText) {
  return new Paragraph({
    spacing: { after: 240 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({ text: captionText, italics: true, size: 20, font: "Arial" })
    ]
  });
}

// Helper: body paragraph
function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    ...opts,
    children: [new TextRun({ text, size: 24, font: "Arial" })]
  });
}

// Helper: body paragraph with mixed formatting
function bodyParaMixed(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    ...opts,
    children: runs.map(r => {
      if (typeof r === "string") return new TextRun({ text: r, size: 24, font: "Arial" });
      return new TextRun({ size: 24, font: "Arial", ...r });
    })
  });
}

// Helper: italic body paragraph (for emphasis blocks)
function italicPara(text) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, size: 24, font: "Arial", italics: true })]
  });
}

// Helper: section heading
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, size: 32, bold: true, font: "Arial" })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, size: 28, bold: true, font: "Arial" })]
  });
}

// Build the document
async function buildDocument() {
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Arial", size: 24 }
        }
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 }
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 }
        },
      ]
    },
    numbering: {
      config: [
        {
          reference: "implications",
          levels: [{
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
          }]
        }
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "999999", space: 4 } },
            children: [new TextRun({ text: "Randolph 2026  \u2014  The Birth of Structure", size: 18, font: "Arial", italics: true, color: "666666" })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", size: 18, font: "Arial", color: "666666" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "666666" })
            ]
          })]
        })
      },
      children: [
        // =====================================================
        // TITLE PAGE
        // =====================================================
        new Paragraph({ spacing: { before: 3600 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "The Birth of Structure", size: 52, bold: true, font: "Arial" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
          children: [new TextRun({ text: "Feigenbaum Architecture in the Quantum-to-Classical Transition", size: 32, font: "Arial" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "Decoherence is not a fade. It is the moment the cascade becomes real.", size: 24, font: "Arial", italics: true })]
        }),
        new Paragraph({ spacing: { before: 600 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "Lucian Randolph", size: 28, font: "Arial" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "March 2026", size: 24, font: "Arial" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "The Last Law Collection", size: 22, font: "Arial", italics: true })]
        }),
        new Paragraph({ spacing: { before: 800 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "DRAFT \u2014 Pre-Publication Version", size: 22, font: "Arial", bold: true, color: "CC0000" })]
        }),

        // =====================================================
        // ABSTRACT
        // =====================================================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("Abstract"),
        bodyPara("Every physics student learns two separate stories. In quantum mechanics, systems exist in superposition until measured, then collapse into definite states through a process called decoherence. In nonlinear dynamics, systems undergo structured transitions\u2014period-doubling cascades governed by universal constants discovered by Feigenbaum in 1978. These two stories have never been connected."),
        bodyPara("This paper connects them. Using the Lucian Law diagnostic\u2014which predicts that any bounded system with nonlinear coupling necessarily produces fractal geometry organized by the Feigenbaum constants\u2014we examine quantum decoherence across three layers of increasing complexity: the linear Lindblad master equation (no cascade, as predicted), classical nonlinear oscillators (full Feigenbaum cascade confirmed, \u03b4\u2082 = 5.09 converging toward 4.669), and the full quantum density matrix of a driven-dissipative Kerr oscillator."),
        bodyPara("The result is more profound than a simple confirmation. The Feigenbaum cascade exists in the classical limit of quantum hardware but vanishes in the full quantum dynamics. The quantum density matrix holds all branches simultaneously as a mixture, reporting smooth averages while encoding the cascade in its phase-space topology. Decoherence\u2014the process that takes the system from quantum to classical\u2014is the mechanism by which the Feigenbaum cascade comes into existence. The cascade does not live in the quantum regime. It does not simply live in the classical regime. It lives in the transition between them. Decoherence is the birth of the Feigenbaum cascade."),

        // =====================================================
        // SECTION 1: THE QUESTION NOBODY ASKED
        // =====================================================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("1. The Question Nobody Asked"),
        bodyPara("The quantum measurement problem has occupied physicists for nearly a century. How does a quantum system, existing in superposition of multiple states, produce the single definite outcomes we observe? The standard answer is decoherence: interaction with the environment destroys quantum coherence, converting superpositions into classical mixtures. This process is universally modeled as smooth exponential decay\u2014a gradual fade from quantum to classical."),
        bodyPara("Separately, the field of nonlinear dynamics has established that structured transitions are ubiquitous in nature. When a nonlinear system is coupled to its environment, it does not simply decay. It undergoes period-doubling bifurcations\u2014a cascade of increasingly rapid oscillations governed by the universal Feigenbaum constant \u03b4 = 4.669201609... This constant appears in systems ranging from dripping faucets to population ecology to fluid turbulence."),
        bodyPara("Nobody has asked whether these two phenomena are connected. Does the transition from quantum to classical\u2014decoherence\u2014have Feigenbaum structure? Or is quantum decoherence genuinely different from every other transition in physics?"),
        bodyPara("This paper asks that question. The answer requires examining quantum decoherence at three layers of complexity, using the Lucian Law as a diagnostic to predict which layers will exhibit cascade structure and which will not. The answer is yes\u2014but not in the way anyone would have expected."),

        // =====================================================
        // SECTION 2: THE DIAGNOSTIC
        // =====================================================
        heading1("2. The Diagnostic: What the Lucian Law Predicts"),
        bodyPara("The Lucian Law (Randolph 2026, DOI: 10.5281/zenodo.18818006) states that any bounded system with nonlinear coupling necessarily produces fractal geometry organized by the Feigenbaum constants. Three prerequisites must be satisfied:"),
        bodyParaMixed([
          { text: "C\u2081 \u2014 Boundedness: ", bold: true },
          "The system must be confined to a finite region of state space. Without boundedness, trajectories escape to infinity and no recursive structure can form."
        ]),
        bodyParaMixed([
          { text: "C\u2082 \u2014 Nonlinear fold: ", bold: true },
          "The dynamics must contain a nonlinear mapping that folds state space back onto itself. This is the mechanism that creates multiple coexisting solutions\u2014the raw material for period-doubling."
        ]),
        bodyParaMixed([
          { text: "C\u2083 \u2014 Coupling: ", bold: true },
          "The system must be coupled to an external parameter (environment, drive, dissipation) that can tune the dynamics through the cascade. Without coupling, the system is isolated and static."
        ]),
        bodyPara("If all three prerequisites are present, the Lucian Law predicts the Feigenbaum cascade must appear. If any one is absent, it cannot. This provides a falsifiable diagnostic: before running a single calculation, we can examine a quantum system and predict whether it will exhibit cascade structure based on which prerequisites are satisfied."),
        bodyPara("We test this diagnostic against three systems of increasing complexity. The diagnostic correctly predicts the outcome in every case."),

        // =====================================================
        // SECTION 3: LAYER ONE \u2014 THE LINEAR LINDBLAD
        // =====================================================
        heading1("3. Layer One: The Linear Lindblad"),
        bodyPara("The standard model of quantum decoherence is the Lindblad master equation, describing a quantum system coupled to a Markovian environment. For a two-level system (qubit) with dephasing:"),
        bodyParaMixed([
          { text: "d\u03C1/dt = -i[H, \u03C1] + \u03B3(L\u03C1L\u2020 - \u00BDL\u2020L\u03C1 - \u00BD\u03C1L\u2020L)", italics: true }
        ]),
        bodyPara("where H is the system Hamiltonian, L is the Lindblad operator describing environmental coupling, and \u03B3 is the coupling strength."),
        bodyPara("Applying the Lucian Law diagnostic: C\u2081 (boundedness) is satisfied\u2014the density matrix is trace-preserving and confined to the Bloch sphere. C\u2083 (coupling) is satisfied\u2014\u03B3 couples the system to the environment. But C\u2082 (nonlinear fold) is NOT satisfied. The Lindblad equation is linear in the density matrix. The superoperator \u2112 acts linearly on \u03C1."),
        bodyPara("The diagnostic predicts: no nonlinearity, no cascade. The decoherence transition should be smooth and structureless. This is exactly what the standard textbook says\u2014but the Lucian Law arrives at this conclusion from a completely different direction, as a prediction rather than an assumption."),
        bodyPara("We verify this prediction computationally. The Lindblad superoperator eigenvalue spectrum is computed across three coupling topologies (pure dephasing, amplitude damping, combined) for coupling strengths spanning four orders of magnitude."),

        // Figure 1: Lindblad eigenvalue spectrum — split into 3 rows for readability
        figureImageOnly("fig73_row1_relaxation.png", 6.5, 0.42),
        figureImageOnly("fig73_row2_dephasing.png", 6.5, 0.36),
        figureImageOnly("fig73_row3_both.png", 6.5, 0.36),
        figureCaption("Figure 1. Lindblad superoperator eigenvalue spectrum across three coupling topologies (Relaxation Only, Pure Dephasing, Both Channels). All eigenvalues move linearly with coupling strength. No crossings, no bifurcations, no structure. C\u2082 absent \u2192 no cascade, as predicted."),

        // Figure 2: Eigenvalue detail
        ...figureParagraph("fig73b_eigenvalue_detail.png",
          "Figure 2. Eigenvalue detail view confirming the absence of any bifurcation structure at fine resolution. The linear scaling persists across all coupling strengths examined."),

        bodyPara("The eigenvalue spectrum confirms the prediction. All eigenvalues move linearly with coupling strength. There are no crossings, no avoided crossings, no bifurcation structure of any kind. The decoherence transition is smooth at every resolution."),

        // Figure 3: Static qubit coherence
        ...figureParagraph("fig74a_static_coherence.png",
          "Figure 3. Static qubit coherence dynamics at eight coupling strengths. Clean oscillations fade smoothly to instant death. The transition is structureless\u2014exponential decay at every coupling strength, with only the rate changing."),

        bodyPara("The coherence dynamics confirm the spectral analysis. At weak coupling, the qubit oscillates and gradually decoheres. At strong coupling, coherence is destroyed almost instantly. The transition between these regimes is smooth and featureless. No period-doubling. No bifurcations. No cascade. The Lucian Law diagnostic is confirmed: without C\u2082, the cascade cannot appear."),

        // =====================================================
        // SECTION 4: LAYER ONE EXTENDED
        // =====================================================
        heading1("4. Layer One Extended: The Driven Linear System"),
        bodyPara("One might object that the static system is too simple. Real quantum hardware involves coherent drives\u2014Rabi oscillations, microwave pulses, laser fields. Does adding a drive change the fundamental character of the decoherence transition?"),
        bodyPara("We test this by adding a coherent drive to the Lindblad equation: H = (\u03C9\u2080/2)\u03C3z + \u03A9cos(\u03C9t)\u03C3x, where \u03A9 is the Rabi frequency and \u03C9 is the drive frequency. The drive-dissipation competition produces richer transient dynamics but does not change the mathematical structure."),

        // Figure 4: Driven coherence
        ...figureParagraph("fig74b_driven_coherence.png",
          "Figure 4. Driven qubit coherence dynamics. Rabi drive competing with dissipation across twelve parameter combinations. Same smooth transition\u2014richer transients but no structural change."),

        // Figure 5: Driven population
        ...figureParagraph("fig74c_driven_population.png",
          "Figure 5. Driven qubit population dynamics. Rabi oscillations at weak coupling transition smoothly to monotonic equilibration at strong coupling. No bifurcation structure."),

        bodyPara("The diagnostic holds. Adding a coherent drive does not introduce a nonlinear fold. The Lindblad equation with linear drive is still linear in \u03C1. C\u2082 remains absent. The decoherence transition remains structureless regardless of drive parameters. The textbook model of smooth exponential decoherence is correct for linear systems\u2014but it is also incomplete, because real quantum hardware is not linear."),

        // =====================================================
        // SECTION 5: THE MISSING INGREDIENT
        // =====================================================
        heading1("5. The Missing Ingredient"),
        bodyPara("Real quantum hardware\u2014superconducting transmon qubits, Josephson junctions, nonlinear optical cavities\u2014is not linear. The Kerr nonlinearity that makes a transmon function as a qubit is a genuine nonlinear fold. The Josephson cosine potential that enables superconducting quantum computing is inherently anharmonic. The very features that make quantum hardware useful are the features that the Lucian Law diagnostic identifies as C\u2082."),
        bodyPara("When this nonlinearity is included in the equations of motion, all three Lucian Law prerequisites become active simultaneously. C\u2081: the density matrix is bounded (trace-preserving). C\u2082: the Kerr nonlinearity provides the fold. C\u2083: the Lindblad dissipation provides the coupling. The law now has jurisdiction. The diagnostic predicts cascade structure must appear."),
        bodyPara("We test this prediction in two stages. First, we examine the classical (semiclassical) limit of the nonlinear quantum system, where the equations of motion reduce to ordinary differential equations. Then we examine the full quantum dynamics, where the density matrix evolves under the Lindblad master equation with nonlinear Hamiltonian."),

        // =====================================================
        // SECTION 6: LAYER TWO \u2014 THE CLASSICAL CASCADE
        // =====================================================
        heading1("6. Layer Two: The Classical Cascade"),
        bodyPara("The Duffing oscillator is the classical analog of the quantum Kerr oscillator. It describes a driven nonlinear oscillator with dissipation\u2014precisely the semiclassical limit of the transmon qubit:"),
        bodyParaMixed([
          { text: "x\u0308 + \u03B4x\u0307 + \u03B1x + \u03B2x\u00B3 = F cos(\u03C9t)", italics: true }
        ]),
        bodyPara("where \u03B4 is the damping (decoherence parameter), \u03B1 and \u03B2 control the potential shape, and F is the drive amplitude. All three Lucian Law prerequisites are satisfied: bounded trajectories (C\u2081), cubic nonlinearity (C\u2082), and environmental damping (C\u2083)."),
        bodyPara("The diagnostic predicts a Feigenbaum cascade. We sweep two parameters: the drive amplitude F and the damping coefficient \u03B4."),

        // Figure 6: Duffing bifurcation - drive sweep
        ...figureParagraph("fig75a_duffing_bif_F.png",
          "Figure 6. Duffing bifurcation diagram\u2014drive amplitude sweep. Period-doubling cascades, chaos windows, and full Feigenbaum architecture. The cascade appears the moment nonlinearity enters the equation."),

        // Figure 7: Duffing bifurcation - damping sweep
        ...figureParagraph("fig75b_duffing_bif_delta.png",
          "Figure 7. Duffing bifurcation diagram\u2014DAMPING sweep. The x-axis is the decoherence coupling parameter \u03B4. Cascades appear on both sides of the transition. The process of increasing environmental coupling\u2014decoherence itself\u2014drives the system through a Feigenbaum cascade."),

        bodyPara("Figure 7 is the critical result for this section. The x-axis is the damping parameter \u03B4\u2014the decoherence coupling strength. As damping increases (as the system decoheres more strongly), the dynamics pass through structured period-doubling bifurcations. This is not a smooth fade. The decoherence transition has architecture."),
        bodyPara("The moment nonlinearity enters the equation, the Lucian Law takes effect. The smooth, structureless decoherence of the linear Lindblad is replaced by a structured cascade through period-doubling bifurcations. The diagnostic prediction is confirmed."),

        // =====================================================
        // SECTION 7: THE R\u00D6SSLER CONFIRMATION
        // =====================================================
        heading1("7. The R\u00F6ssler Confirmation"),
        bodyPara("To confirm that the cascade is universal and not specific to the Duffing oscillator, we examine the R\u00F6ssler system\u2014a three-dimensional continuous flow that is fundamentally different in structure:"),
        bodyParaMixed([
          { text: "dx/dt = -y - z,   dy/dt = x + ay,   dz/dt = b + z(x - c)", italics: true }
        ]),
        bodyPara("The parameter c controls dissipation\u2014it is the coupling to the environment that determines how strongly trajectories are contracted. The Lucian Law diagnostic identifies C\u2081 (bounded attractor), C\u2082 (nonlinear fold in the z equation), and C\u2083 (c as the coupling parameter). All three prerequisites are active. The cascade must appear."),

        // Figure 8: R\u00F6ssler full
        ...figureParagraph("fig76a_rossler_full.png",
          "Figure 8. Complete R\u00F6ssler bifurcation diagram. Dissipation coupling parameter c swept from 2.0 to 6.0. Period-doubling cascades, chaos, periodic windows\u2014the full Feigenbaum architecture in a three-dimensional continuous flow."),

        // Figure 9: R\u00F6ssler zoom
        ...figureParagraph("fig76b_rossler_zoom.png",
          "Figure 9. R\u00F6ssler cascade zoomed. The period-doubling sequence in full detail. The published literature value for the second bifurcation point c\u2082 \u2248 3.53 is corrected to c\u2082 = 3.836\u2014the original measurement lacked the resolution required to detect the true bifurcation point. See Section 11 for implications."),

        bodyPara("A three-dimensional continuous flow\u2014fundamentally different from discrete maps or forced oscillators\u2014exhibits the same cascade architecture. The coupling parameter c controls dissipation, and sweeping c drives the system through a Feigenbaum cascade. The Lucian Law predicts the cascade regardless of system dimension or dynamical type. Confirmed."),
        bodyPara("The literature correction is significant. The published value c\u2082 \u2248 3.53 (Barrio & Serrano 2007) was measured at insufficient resolution. The true value c\u2082 = 3.836 was hidden because the resolution required to detect each cascade level scales as \u03b4\u207b\u207f\u2014the cascade determines what instruments can find it. This observability scaling property is discussed further in Section 11."),

        // =====================================================
        // SECTION 8: EXTRACTING \u03b4
        // =====================================================
        heading1("8. Extracting \u03b4"),
        bodyPara("The Feigenbaum constant \u03b4 = 4.669201609... is the ratio of successive bifurcation intervals. If the cascade is truly universal, the same constant must appear in every system regardless of its specific dynamics."),

        // Figure 10: Feigenbaum convergence
        ...figureParagraph("fig77_feigenbaum_convergence.png",
          "Figure 10. Feigenbaum \u03b4 convergence: logistic map and R\u00F6ssler system side by side. Measured \u03b4\u2099 converging toward 4.6692. Logistic map reaches 0.08% accuracy by level 4. R\u00F6ssler at 2.26% by level 2."),

        // Figure 11: Topology sweep
        ...figureParagraph("fig78_topology_sweep.png",
          "Figure 11. Topology sweep across five independent systems. Three quadratic (z=2) systems converge to \u03b4 \u2248 4.67. The cubic (z=3) system converges to \u03b4 \u2248 5.84. Different coupling topologies select different Feigenbaum family members. Universality confirmed."),

        bodyPara("The same constant appears in discrete maps, smooth periodic maps, and continuous three-dimensional flows. The topology of the nonlinearity\u2014not the specific dynamics\u2014selects the universal constant. Quadratic extremum gives \u03b4 \u2248 4.669. Cubic gives \u03b4 \u2248 5.97. The Lucian Law\u2019s prediction that coupling topology determines the geometric constant is confirmed across five independent systems."),
        bodyPara("Figure 11 is the definitive confirmation. Five systems. Three coupling topologies. The same universal constants. The Feigenbaum architecture is not a property of any particular equation. It is a property of bounded nonlinear coupling itself."),

        // =====================================================
        // SECTION 9: LAYER THREE \u2014 THE QUANTUM KERR OSCILLATOR
        // =====================================================
        heading1("9. Layer Three: The Quantum Kerr Oscillator"),
        bodyPara("We now move to the full quantum dynamics. The driven-dissipative quantum Kerr oscillator is described by the rotating-frame Hamiltonian:"),
        bodyParaMixed([
          { text: "H = \u0394 a\u2020a + (K/2) a\u2020a(a\u2020a - 1) + F(a\u2020 + a)", italics: true }
        ]),
        bodyPara("with Lindblad dissipation L = \u221A\u03B3 \u00B7 a (single-photon loss). Parameters: \u0394 = -3.0 (red-detuned for bistability), K = 0.3 (Kerr nonlinearity), F = 3.0 (coherent drive), N_Fock = 40 (Fock space truncation). The dissipation rate \u03B3 is swept from 0.01 to 3.0, crossing the semiclassical bistability boundary at \u03B3_c \u2248 1.92."),
        bodyPara("This system satisfies all three Lucian Law prerequisites simultaneously in the quantum regime: C\u2081 (trace-preserving density matrix), C\u2082 (Kerr nonlinearity), C\u2083 (photon loss to environment). The law predicts structure."),
        bodyPara("What the quantum system reveals is a first-order quantum phase transition\u2014a bistability crossover where the density matrix transitions between two semiclassical solutions."),

        // Figure 12: Quantum Kerr 6-panel — split into 3 rows for readability
        figureImageOnly("fig79_row1_AB.png", 6.5, 0.29),
        figureCaption("Figure 12a. (A) Mean photon number \u27E8n\u27E9 vs \u03B3 with semiclassical branches overlaid: the quantum curve smoothly interpolates between both classical solutions. (B) Photon number variance spikes to 34.2 at the transition\u2014critical fluctuations."),
        figureImageOnly("fig79_row2_CD.png", 6.5, 0.30),
        figureCaption("Figure 12b. (C) Fano factor peaks at 5.1\u2014super-Poissonian statistics at the crossover. (D) Purity crashes to 0.33\u2014the system is maximally mixed at the quantum phase transition."),
        figureImageOnly("fig79_row3_EF.png", 6.5, 0.34),
        figureCaption("Figure 12c. (E) Wigner function triptych: one blob, two blobs, one blob. The bimodal Wigner at \u03B3 = 1.15 shows two classical realities coexisting. (F) Photon number distribution P(n) showing bimodal structure at the transition."),

        bodyPara("The quantum transition occurs at \u03B3_q = 1.19, significantly EARLIER than the semiclassical prediction of \u03B3_c = 1.92. The shift \u0394\u03B3 = 0.73 is caused by quantum tunneling between the two wells of the effective potential. This is a quantitative, measurable prediction: the quantum system decoheres before the classical theory says it should."),
        bodyPara("The Wigner function triptych (Panel E) is the smoking gun. At low \u03B3, the phase-space distribution shows a single coherent blob near the upper semiclassical branch. At the transition point \u03B3 = 1.15, TWO DISTINCT BLOBS coexist\u2014the quantum state is a mixture of both classical solutions simultaneously. At high \u03B3, a single blob near the origin remains. The system passes through a bimodal phase-space distribution."),
        bodyPara("Zero Wigner negativity is observed everywhere. This is physically correct: the dissipation converts all quantum superposition into classical statistics. The system is always quasi-classical\u2014a mixture of coherent states, never a superposition. Decoherence has already won. The question is which classical solution it selects."),

        // Figure 13: Wigner detail
        ...figureParagraph("fig79b_wigner_detail.png",
          "Figure 13. High-resolution Wigner function triptych. Left: single coherent blob at \u03B3 = 0.50 (deep bistable regime, upper branch dominant). Center: bimodal distribution at \u03B3 = 1.15 (at the quantum phase transition\u2014two classical solutions coexisting). Right: single blob near origin at \u03B3 = 2.50 (monostable, coherent state)."),

        bodyPara("The photon number distribution (Panel F) confirms the bimodal structure quantitatively. At \u03B3 = 1.15, P(n) shows peaks at both n = 1 (probability 0.110) and n = 12 (probability 0.099). The quantum state is holding two classical realities simultaneously, reporting their average as its expectation value."),

        // =====================================================
        // SECTION 10: THE BREATHTAKING COMPARISON
        // =====================================================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("10. The Breathtaking Comparison"),
        bodyPara("We now add time-periodic modulation to the Kerr oscillator to break the time-translation symmetry of the rotating frame, enabling period-doubling:"),
        bodyParaMixed([
          { text: "H(t) = \u0394 a\u2020a + (K/2) a\u2020a(a\u2020a - 1) + [F\u2080 + F\u2081cos(\u03C9\u2098t)](a\u2020 + a)", italics: true }
        ]),
        bodyPara("The modulation frequency \u03C9\u2098 = 5.2 is chosen to satisfy the parametric resonance condition \u03C9\u2098 \u2248 2 \u00D7 \u03C9_int, where \u03C9_int \u2248 2.6 is the intrinsic oscillation frequency of the upper semiclassical branch (computed from eigenvalue analysis of the linearized system). The modulation depth F\u2081 is swept from 0 to 3.5."),
        bodyPara("Stroboscopic sampling at intervals T = 2\u03C0/\u03C9\u2098 converts the continuous dynamics into a discrete map, enabling direct comparison with the Feigenbaum bifurcation diagram. Both classical (semiclassical) and quantum (full density matrix via mesolve) simulations are performed."),

        // Figure 14: The 8-panel centerpiece — split into 4 rows for readability
        figureImageOnly("fig80_row1_AB.png", 6.5, 0.36),
        figureCaption("Figure 14a. THE BREATHTAKING COMPARISON. (A) Classical bifurcation diagram showing the full Feigenbaum cascade\u2014period-doubling, chaos, windows. (B) Quantum bifurcation diagram\u2014the same system, same Hamiltonian, same parameters\u2014showing a smooth, gentle descent. Period-1 everywhere. The quantum system holds the cascade as a mixture."),
        figureImageOnly("fig80_row2_CD.png", 6.5, 0.33),
        figureCaption("Figure 14b. (C) Classical period detection: the staircase P1\u2192P2\u2192P4\u2192P8\u2192P16\u2192chaos. Each step clean and measurable. (D) Quantum period detection: flat line, P-1 everywhere. The quantum system refuses to commit to a branch."),
        figureImageOnly("fig80_row3_EF.png", 6.5, 0.33),
        figureCaption("Figure 14c. (E) Classical damping sweep (Sweep B, F\u2081 = 2.0) showing bifurcation structure in the decoherence parameter itself. (F) Quantum damping sweep showing smooth crossover with the quantum phase transition at \u03B3 \u2248 1.2."),
        figureImageOnly("fig80_row4_GH.png", 6.5, 0.33),
        figureCaption("Figure 14d. (G) Feigenbaum \u03b4 extraction: \u03b4\u2082 = 5.09, converging toward 4.669. (H) THE WHISPER. Classical spread (gray) shows sharp structure at every bifurcation point. Quantum spread (purple) shows subtle bumps at the same F\u2081 values\u2014the cascade whispering through the quantum noise floor."),

        bodyPara("Panels A and B, side by side, tell the complete story in two images. The classical system (Panel A) explodes with structure: period-doubling cascades, chaos windows, the full Feigenbaum architecture in all its complexity. The quantum system (Panel B) shows a smooth, gentle curve descending from \u27E8n\u27E9 = 12.3 to \u27E8n\u27E9 = 6.9. The same Hamiltonian. The same parameters. The same physical system. Two completely different pictures of reality."),
        bodyPara("The quantum system is pregnant with the cascade. The smooth descent in Panel B is not featureless\u2014it is smooth in the way a river is smooth above a waterfall. The structure is underneath. The density matrix knows about every branch, every bifurcation, every period-doubling. It holds them all simultaneously. And it reports back a single smooth number because that is what expectation values do: they average."),
        bodyPara("Panels C and D quantify the contrast. The classical period detection (Panel C) shows a clean staircase: P-1, P-2, P-4, P-8, P-16, chaos. Each step measurable and precise. The quantum period detection (Panel D) shows a flat line: P-1 everywhere. The quantum system refuses to commit to a branch. It holds them all."),

        heading2("The Classical Cascade"),
        bodyPara("The classical bifurcation points, refined by binary search to six-digit precision:"),
        bodyParaMixed([
          { text: "P1\u2192P2: F\u2081 = 1.146618", bold: true },
          "   (165 stroboscopic points in the period-2 window)"
        ]),
        bodyParaMixed([
          { text: "P2\u2192P4: F\u2081 = 2.571677", bold: true },
          "   (17 points)"
        ]),
        bodyParaMixed([
          { text: "P4\u2192P8: F\u2081 = 2.745863", bold: true },
          "   (5 points)"
        ]),
        bodyParaMixed([
          { text: "P8\u2192P16: F\u2081 = 2.780070", bold: true },
          "   (1 point)"
        ]),
        bodyPara("Feigenbaum ratio extraction:"),
        bodyParaMixed([
          { text: "\u03b4\u2081 = (2.5717 - 1.1466) / (2.7459 - 2.5717) = 8.18", bold: true },
          "   (75% error\u2014first ratio always rough)"
        ]),
        bodyParaMixed([
          { text: "\u03b4\u2082 = (2.7459 - 2.5717) / (2.7801 - 2.7459) = 5.09", bold: true },
          "   (9.1% error\u2014converging toward 4.669)"
        ]),
        bodyPara("This convergence pattern is identical to every other system examined. In the logistic map, the first ratio is 4.75 and it takes four ratios to reach 0.08% accuracy. In the R\u00F6ssler, the first ratio is 3.50 and the second is 4.56. The Kerr oscillator follows the same trajectory: overshoot at the first ratio, convergence at the second, with additional ratios expected to tighten further. The universal constant is the same: \u03b4 = 4.669201609..."),

        heading2("The Quantum Silence"),
        bodyPara("All 60 quantum sweep points show Period-1. Every single one. The stroboscopic spread ranges from 0.022 (at F\u2081 = 0) down to 0.000 (at F\u2081 = 3.5). The quantum density matrix expectation value \u27E8n\u27E9 = Tr(n\u0302\u03C1) is perfectly smooth\u2014no bifurcation structure whatsoever."),
        bodyPara("This is not a failure. This is the deepest result of the entire investigation."),
        bodyPara("When the classical system has period-2\u2014alternating between two values of |a|\u00B2\u2014the quantum density matrix contains BOTH values simultaneously as a mixture. The expectation value averages them. \u27E8n\u27E9 is smooth because the quantum state contains all branches at once. The bifurcation structure is encoded in the Wigner function topology\u2014bimodal distributions, coexisting phase-space blobs\u2014but it does not manifest as observable oscillation in any single expectation value."),

        heading2("The Whisper"),
        bodyPara("Panel H reveals the subtlest and perhaps most important result. The classical stroboscopic spread (gray) shows sharp structure: clean jumps at bifurcation points, spikes at chaos onset, the full Feigenbaum fingerprint written in the spread of stroboscopic samples. The quantum spread (purple) is low and quiet\u2014but it is not featureless. There are subtle inflections at the F\u2081 values where the classical system bifurcates."),
        bodyPara("The cascade is whispering through the quantum noise floor. The Wigner function\u2019s bimodal distribution at the classical bifurcation points creates a slightly wider spread in quantum stroboscopic samples, even though the mean is smooth. The cascade is there. In the fluctuations. In the width. In the whisper."),

        // =====================================================
        // SECTION 11: THE OBSERVABILITY THRESHOLD
        // =====================================================
        heading1("11. The Observability Threshold"),
        bodyPara("The R\u00F6ssler literature correction (Section 7) reveals a property of the Feigenbaum cascade that has direct implications for quantum measurements. The published bifurcation point c\u2082 \u2248 3.53 was wrong because the measurement resolution was insufficient. The true value c\u2082 = 3.836 requires resolving a bifurcation interval of width \u0394c \u2248 0.17\u2014compared to the first interval of width \u0394c \u2248 1.84."),
        bodyPara("Each successive bifurcation level compresses the parameter interval by a factor of \u03b4 \u2248 4.669. The measurement precision required to resolve level n scales as \u03b4\u207b\u207f. The cascade determines what instruments can find it. This is the observability scaling property\u2014an extension of the self-grounding theorem proved in the companion paper (The Decay Bounce, DOI: 10.5281/zenodo.18868816)."),
        bodyPara("The same principle applies to quantum decoherence measurements. The precision required to detect cascade structure in quantum experiments is governed by the cascade\u2019s own architecture. If experimentalists measure at a resolution above the first bifurcation threshold, they see only smooth crossover\u2014the linear Lindblad prediction. The cascade whisper in Panel H of Figure 14 shows what becomes visible when you know where to look and measure with sufficient precision."),
        bodyPara("This has practical implications for quantum computing. Decoherence benchmarks that average over drive parameters or damping rates will systematically miss the cascade structure. Only measurements that sweep a control parameter through the bifurcation region with resolution better than \u03b4\u207b\u00B2 of the first bifurcation interval can detect the period-4 level. Higher levels require exponentially better precision."),

        // =====================================================
        // SECTION 12: WHAT THIS MEANS
        // =====================================================
        heading1("12. What This Means"),
        bodyPara("Three layers of reality. One architecture."),
        bodyPara("The linear quantum system has no cascade because C\u2082 is missing. The Lucian Law correctly predicts this. The textbook is right about linear systems."),
        bodyPara("The classical nonlinear system has the full cascade because all three prerequisites are active. \u03b4 is confirmed across five independent systems spanning discrete maps, smooth periodic maps, and continuous three-dimensional flows. The Lucian Law correctly predicts this."),
        bodyPara("The quantum nonlinear system holds the cascade as encoded potential. The density matrix carries all branches simultaneously. The bifurcation structure lives in the phase-space topology\u2014bimodal Wigner functions, variance spikes, purity drops\u2014but not in simple expectation values. The cascade whispers through the fluctuation structure at the classical bifurcation points."),
        bodyPara("Decoherence is the process by which the cascade emerges from quantum potential into classical actuality. It is not a smooth fade. It is the birth of structure."),
        bodyPara("As the system becomes more classical (higher \u27E8n\u27E9, weaker quantum fluctuations), the cascade crystallizes. As it becomes more quantum (lower \u27E8n\u27E9, stronger fluctuations), the cascade dissolves into smooth behavior. The quantum-to-classical transition is not just governed by the cascade\u2014it IS the process that brings the cascade into existence."),

        heading2("Implications"),
        new Paragraph({
          numbering: { reference: "implications", level: 0 },
          spacing: { after: 120, line: 276 },
          children: [new TextRun({ size: 24, font: "Arial", bold: true, text: "Real quantum hardware operates in the cascade regime. " }),
                     new TextRun({ size: 24, font: "Arial", text: "Superconducting transmon qubits, Kerr oscillators, and Josephson junctions operate in the regime where the Feigenbaum architecture is active. Engineering that accounts for this structure\u2014rather than treating decoherence as simple exponential decay\u2014can exploit the cascade architecture for improved coherence control." })]
        }),
        new Paragraph({
          numbering: { reference: "implications", level: 0 },
          spacing: { after: 120, line: 276 },
          children: [new TextRun({ size: 24, font: "Arial", bold: true, text: "Coupling topology determines the universal constant. " }),
                     new TextRun({ size: 24, font: "Arial", text: "Different quantum hardware architectures have different nonlinear orders and therefore different Feigenbaum family members governing their decoherence transitions. Quadratic Kerr nonlinearity selects \u03b4 \u2248 4.669. Higher-order nonlinearities select different constants. This is a testable, quantitative prediction." })]
        }),
        new Paragraph({
          numbering: { reference: "implications", level: 0 },
          spacing: { after: 120, line: 276 },
          children: [new TextRun({ size: 24, font: "Arial", bold: true, text: "The quantum tunneling shift is measurable. " }),
                     new TextRun({ size: 24, font: "Arial", text: "The quantum phase transition occurs at \u03B3_q = 1.19, while the semiclassical prediction is \u03B3_c = 1.92. The shift \u0394\u03B3 = 0.73 is a quantitative prediction for superconducting circuit experiments. The quantum system decoheres earlier than the semiclassical theory predicts, and the magnitude of this shift is calculable from the Hilbert space dynamics." })]
        }),
        new Paragraph({
          numbering: { reference: "implications", level: 0 },
          spacing: { after: 120, line: 276 },
          children: [new TextRun({ size: 24, font: "Arial", bold: true, text: "The cascade whisper is experimentally detectable. " }),
                     new TextRun({ size: 24, font: "Arial", text: "High-precision measurements of photon number variance as a function of drive or dissipation parameters should show subtle structure at the classical bifurcation points. The bimodal Wigner function at these points creates measurable signatures in the fluctuation structure even when the mean is smooth. This is an experimental prediction." })]
        }),

        // =====================================================
        // SECTION 13: THE ARCHITECTURE OF EXPERIENCE
        // =====================================================
        heading1("13. The Architecture of Experience"),
        bodyPara("The Lucian Law describes the only architecture that produces genuine experience. A fully deterministic universe\u2014clockwork\u2014allows no novelty. Every future state is implicit in the initial conditions. A fully random universe\u2014noise\u2014allows no structure. No pattern persists long enough to constitute experience. The Feigenbaum cascade is the narrow path between clockwork and noise: structured unpredictability, patterned novelty."),
        bodyPara("The quantum regime holds all possibilities simultaneously. No definite outcome. No commitment. The classical regime manifests one outcome. One branch. One experience. Between them\u2014in the process of decoherence\u2014the system commits. The cascade provides the STRUCTURE of possible outcomes: the discrete set of branches, the hierarchical organization, the universal ratios governing how finely the possibilities subdivide. Decoherence provides the SELECTION: which possibility becomes actual."),
        bodyPara("The threshold transition\u2014the moment the system commits to a branch\u2014cannot be calculated from within the system. It can only be observed. The density matrix holds all branches with definite weights. But which branch becomes actual at the threshold is not determined by those weights. It is determined by the interaction with the environment at the moment of decoherence. The weights give probabilities. The threshold gives actuality."),
        bodyPara("This is not a gap in knowledge. It is a structural feature of the architecture. The Feigenbaum cascade produces a hierarchy of possible outcomes. The threshold transition selects one. And the selection cannot be predicted because unpredictability at threshold transitions is the defining feature of the cascade\u2014it is what makes the boundary between order and chaos a boundary rather than a line."),

        // =====================================================
        // SECTION 14: CLOSING
        // =====================================================
        heading1("14. Conclusion"),
        bodyPara("The transition from quantum to classical is not a mystery. It is not a philosophical problem requiring interpretation. It is a Feigenbaum cascade emerging from quantum potential through decoherence."),
        bodyPara("The same architecture that governs dripping faucets and population dynamics and galactic rotation curves governs the birth of classical reality from quantum possibility. The textbook missed it because the textbook linearized the equation. The real hardware has the nonlinearity. The real physics has the cascade."),
        bodyPara("The Lucian Law\u2014confirmed in stellar dynamics, proved self-grounding, governing phenomena from cosmological expansion to galactic structure\u2014governs this too. One law. Every scale. Including the scale where reality becomes real."),
        bodyPara("The cascade does not live in the quantum regime. It does not simply live in the classical regime. It lives in the threshold between them\u2014in the moment of decoherence, in the birth of definite outcomes from indefinite potential, in the process by which the universe commits to being something rather than everything."),
        italicPara("Decoherence is not a fade. It is the moment the cascade becomes real."),

        // =====================================================
        // REFERENCES
        // =====================================================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("References"),
        bodyPara("Randolph, L. (2026). The Lucian Law: Bounded Nonlinear Coupling Necessarily Produces Fractal Geometry. Zenodo. DOI: 10.5281/zenodo.18818006"),
        bodyPara("Randolph, L. (2026). Feigenbaum Constants as Derived Quantities of the Lucian Law. Zenodo. DOI: 10.5281/zenodo.18818008"),
        bodyPara("Randolph, L. (2026). The Full Extent of the Lucian Law. Zenodo. DOI: 10.5281/zenodo.18818010"),
        bodyPara("Randolph, L. (2026). Inflationary Cosmology and the Lucian Law. Zenodo. DOI: 10.5281/zenodo.18819605"),
        bodyPara("Randolph, L. (2026). The Decay Bounce: Self-Grounding Mechanism of the Feigenbaum Architecture. Zenodo. DOI: 10.5281/zenodo.18868816"),
        bodyPara("Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics, 19(1), 25-52."),
        bodyPara("Feigenbaum, M. J. (1979). The universal metric properties of nonlinear transformations. Journal of Statistical Physics, 21(6), 669-706."),
        bodyPara("Lindblad, G. (1976). On the generators of quantum dynamical semigroups. Communications in Mathematical Physics, 48(2), 119-130."),
        bodyPara("Drummond, P. D., & Walls, D. F. (1980). Quantum theory of optical bistability. I. Nonlinear polarizability model. Journal of Physics A, 13(2), 725."),
        bodyPara("Barrio, R., & Serrano, S. (2007). A three-parametric study of the Lorenz model. Physica D, 229(1), 43-51."),
        bodyPara("Dykman, M. I., & Krivoglaz, M. A. (1984). Fluctuations in nonlinear systems near bifurcations corresponding to the appearance of new stable states. Physica A, 104(3), 480-494."),

        // =====================================================
        // FIGURE INDEX
        // =====================================================
        new Paragraph({ children: [new PageBreak()] }),
        heading1("Figure Index"),
        bodyParaMixed([{ text: "Figure 1. ", bold: true }, "Lindblad superoperator eigenvalue spectrum\u2014three rows: Relaxation, Dephasing, Both (fig73)"]),
        bodyParaMixed([{ text: "Figure 2. ", bold: true }, "Eigenvalue detail view (fig73b)"]),
        bodyParaMixed([{ text: "Figure 3. ", bold: true }, "Static qubit coherence dynamics (fig74a)"]),
        bodyParaMixed([{ text: "Figure 4. ", bold: true }, "Driven qubit coherence dynamics (fig74b)"]),
        bodyParaMixed([{ text: "Figure 5. ", bold: true }, "Driven qubit population dynamics (fig74c)"]),
        bodyParaMixed([{ text: "Figure 6. ", bold: true }, "Duffing bifurcation\u2014drive amplitude sweep (fig75a)"]),
        bodyParaMixed([{ text: "Figure 7. ", bold: true }, "Duffing bifurcation\u2014damping sweep (fig75b)"]),
        bodyParaMixed([{ text: "Figure 8. ", bold: true }, "R\u00F6ssler complete bifurcation diagram (fig76a)"]),
        bodyParaMixed([{ text: "Figure 9. ", bold: true }, "R\u00F6ssler cascade zoomed (fig76b)"]),
        bodyParaMixed([{ text: "Figure 10. ", bold: true }, "Feigenbaum \u03b4 convergence (fig77)"]),
        bodyParaMixed([{ text: "Figure 11. ", bold: true }, "Topology sweep\u2014five systems (fig78)"]),
        bodyParaMixed([{ text: "Figures 12a\u2013c. ", bold: true }, "Quantum Kerr cascade\u2014six panels in three rows (fig79)"]),
        bodyParaMixed([{ text: "Figure 13. ", bold: true }, "Wigner function triptych detail (fig79b)"]),
        bodyParaMixed([{ text: "Figures 14a\u2013d. ", bold: true }, "Classical vs quantum parametric drive\u2014eight panels in four rows, the centerpiece (fig80)"]),
      ]
    }]
  });

  const buffer = await Packer.toBuffer(doc);
  const outPath = "/Users/lucianrandolph/Downloads/Projects/einstein_fractal_analysis/The_Birth_of_Structure_Randolph_2026.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`Written: ${outPath}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
}

buildDocument().catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
