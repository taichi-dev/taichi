// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPBENCHSTYLE_H
#define SPBENCHSTYLE_H

void printBenchStyle(std::ofstream& out)
{
  out << "<xsl:stylesheet id='stylesheet' version='1.0' \
      xmlns:xsl='http://www.w3.org/1999/XSL/Transform' >\n \
      <xsl:template match='xsl:stylesheet' />\n \
      <xsl:template match='/'> <!-- Root of the document -->\n \
      <html>\n \
        <head> \n \
          <style type='text/css'> \n \
            td { white-space: nowrap;}\n \
          </style>\n \
        </head>\n \
        <body>";
  out<<"<table border='1' width='100%' height='100%'>\n \
        <TR> <!-- Write the table header -->\n \
        <TH>Matrix</TH> <TH>N</TH> <TH> NNZ</TH>  <TH> Sym</TH>  <TH> SPD</TH> <TH> </TH>\n \
          <xsl:for-each select='BENCH/AVAILSOLVER/SOLVER'>\n \
            <xsl:sort select='@ID' data-type='number'/>\n \
            <TH>\n \
              <xsl:value-of select='TYPE' />\n \
              <xsl:text></xsl:text>\n \
              <xsl:value-of select='PACKAGE' />\n \
              <xsl:text></xsl:text>\n \
            </TH>\n \
          </xsl:for-each>\n \
        </TR>";
        
  out<<"  <xsl:for-each select='BENCH/LINEARSYSTEM'>\n \
          <TR> <!-- print statistics for one linear system-->\n \
            <TH rowspan='4'> <xsl:value-of select='MATRIX/NAME' /> </TH>\n \
            <TD rowspan='4'> <xsl:value-of select='MATRIX/SIZE' /> </TD>\n \
            <TD rowspan='4'> <xsl:value-of select='MATRIX/ENTRIES' /> </TD>\n \
            <TD rowspan='4'> <xsl:value-of select='MATRIX/SYMMETRY' /> </TD>\n \
            <TD rowspan='4'> <xsl:value-of select='MATRIX/POSDEF' /> </TD>\n \
            <TH> Compute Time </TH>\n \
            <xsl:for-each select='SOLVER_STAT'>\n \
              <xsl:sort select='@ID' data-type='number'/>\n \
              <TD> <xsl:value-of select='TIME/COMPUTE' /> </TD>\n \
            </xsl:for-each>\n \
          </TR>";
  out<<"  <TR>\n \
            <TH> Solve Time </TH>\n \
            <xsl:for-each select='SOLVER_STAT'>\n \
              <xsl:sort select='@ID' data-type='number'/>\n \
              <TD> <xsl:value-of select='TIME/SOLVE' /> </TD>\n \
            </xsl:for-each>\n \
          </TR>\n \
          <TR>\n \
            <TH> Total Time </TH>\n \
            <xsl:for-each select='SOLVER_STAT'>\n \
              <xsl:sort select='@ID' data-type='number'/>\n \
              <xsl:choose>\n \
                <xsl:when test='@ID=../BEST_SOLVER/@ID'>\n \
                  <TD style='background-color:red'> <xsl:value-of select='TIME/TOTAL' />  </TD>\n \
                </xsl:when>\n \
                <xsl:otherwise>\n \
                  <TD>  <xsl:value-of select='TIME/TOTAL' /></TD>\n \
                </xsl:otherwise>\n \
              </xsl:choose>\n \
            </xsl:for-each>\n \
          </TR>";
  out<<"  <TR>\n \
              <TH> Error </TH>\n \
              <xsl:for-each select='SOLVER_STAT'>\n \
                <xsl:sort select='@ID' data-type='number'/>\n \
                <TD> <xsl:value-of select='ERROR' />\n \
                <xsl:if test='ITER'>\n \
                  <xsl:text>(</xsl:text>\n \
                  <xsl:value-of select='ITER' />\n \
                  <xsl:text>)</xsl:text>\n \
                </xsl:if> </TD>\n \
              </xsl:for-each>\n \
            </TR>\n \
          </xsl:for-each>\n \
      </table>\n \
    </body>\n \
    </html>\n \
  </xsl:template>\n \
  </xsl:stylesheet>\n\n";
  
}

#endif
