// taken from https://typst.app/project/ruyA4kPNzRyNmyu3MZqxny

// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", authors: (), date: none, body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center, flipped: true, margin: 3em)
  set text(font: "Libertinus Serif", lang: "en", size: 10pt)

  // Set paragraph spacing.
  set par(spacing: 0.25em)

  set heading(numbering: "1.1")
  set par(leading: 0.58em)

  set par(
    first-line-indent: 0.5em,
    spacing: 1em,
    justify: true,
  )
  
  show: columns.with(3, gutter: 1em)

  // Title row.
  align(left)[
    #block(text(weight: 700, 1.75em, title))
    #v(0.8em, weak: true)
    #date, GITCOMMIT, licensed under CC BY-SA 4.0
    #block(authors.map(a => a.name + " (" + a.email + ")").join())
  ]

  // Main body.
  set par(justify: true)
  //set text(size: 0.85em)

  body
}

#let colorbox(title: none, inline: true, breakable: true, color: blue, content) = {
  let colorOutset = 4pt
  let titleContent = if title != none {
    box(
      fill: silver,
      outset: (left: colorOutset - 1pt, rest: colorOutset),
      width: if inline { auto } else { 100% },
      radius: if inline { (bottom-right: 8pt) } else { 0pt },
      [*#title*]) + if inline { h(6pt) }
  }

  block(
    stroke: (left: 2pt + color),
    outset: colorOutset, 
    fill: silver.lighten(60%), 
    breakable: breakable,
    width: 100%,
    
    titleContent + content)
}

#let slashCircle = symbol("\u{2298}")
