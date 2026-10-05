```drawio
<mxfile host="">
  <diagram id="page-1" name="Page-1">
    <mxGraphModel dx="888" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="NAadGAteFJgWO_cu3Jsl-1" edge="1" parent="1" source="NAadGAteFJgWO_cu3Jsl-2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" target="NAadGAteFJgWO_cu3Jsl-8">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-2" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;" value="&lt;font style=&quot;font-size: 8px;&quot;&gt;GroundTruthCalendar&lt;/font&gt;" vertex="1">
          <mxGeometry height="40" width="160" x="18" y="68.33" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-3" parent="1" style="text;whiteSpace=wrap;html=1;fontSize=9;" value="&lt;div style=&quot;background-color:#ffffff;color:#080808&quot;&gt;&lt;pre style=&quot;font-family: &amp;quot;JetBrains Mono&amp;quot;, monospace;&quot;&gt;&lt;span style=&quot;color:#000000;&quot;&gt;frequency&lt;/span&gt;: &lt;span style=&quot;color:#000000;&quot;&gt;pd.Timedelta&lt;/span&gt;,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;tickers_raw_price_data&lt;/span&gt;: &lt;span style=&quot;color:#000000;&quot;&gt;pd.DataFrame&lt;/span&gt;,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;tickers&lt;/span&gt;: &lt;span style=&quot;color:#000080;&quot;&gt;list&lt;/span&gt;[&lt;span style=&quot;color:#000080;&quot;&gt;str&lt;/span&gt;],&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;weights&lt;/span&gt;: &lt;span style=&quot;color:#000080;&quot;&gt;dict&lt;/span&gt;[&lt;span style=&quot;color:#000080;&quot;&gt;str&lt;/span&gt;, &lt;span style=&quot;color:#000080;&quot;&gt;float&lt;/span&gt;] | pd.Series,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;train_start&lt;/span&gt;,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;train_end&lt;/span&gt;,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;forecast_start&lt;/span&gt;,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;forecast_end&lt;/span&gt;,&lt;br&gt;&lt;span style=&quot;color:#000000;&quot;&gt;drop_market_closed&lt;/span&gt;: &lt;span style=&quot;color:#000000;&quot;&gt;bool &lt;/span&gt;= &lt;span style=&quot;color:#0033b3;&quot;&gt;True&lt;/span&gt;,&lt;/pre&gt;&lt;/div&gt;" vertex="1">
          <mxGeometry height="130" width="220" x="21" y="108.32999999999998" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-4" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;autosize=1;resizable=0;fontStyle=1" value="train_start : day 1&lt;div&gt;train_end: day 10&amp;nbsp;&lt;/div&gt;&lt;div&gt;&lt;span style=&quot;background-color: transparent; color: light-dark(rgb(0, 0, 0), rgb(255, 255, 255));&quot;&gt;forecast_start: day 11&lt;/span&gt;&lt;/div&gt;&lt;div&gt;forecast_end = day 15&amp;nbsp;&lt;/div&gt;" vertex="1">
          <mxGeometry height="70" width="150" x="42" y="230" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-5" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;autosize=1;resizable=0;fontStyle=1" value="window_size:" vertex="1">
          <mxGeometry height="30" width="90" x="388" y="273.34000000000003" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-6" edge="1" parent="1" source="NAadGAteFJgWO_cu3Jsl-8" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" target="NAadGAteFJgWO_cu3Jsl-9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-7" edge="1" parent="1" source="NAadGAteFJgWO_cu3Jsl-8" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" target="NAadGAteFJgWO_cu3Jsl-10">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="442" y="88" />
              <mxPoint x="442" y="20" />
              <mxPoint x="753" y="20" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-8" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;" value="ML FEATURE BUILDERS" vertex="1">
          <mxGeometry height="60" width="120" x="292" y="58.33" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-9" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;" value="Training Feautre" vertex="1">
          <mxGeometry height="60" width="160" x="478" y="40" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-10" parent="1" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;" value="Forecast Feature" vertex="1">
          <mxGeometry height="60" width="160" x="673" y="40" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-11" connectable="0" parent="1" style="group;fontSize=9;" value="" vertex="1">
          <mxGeometry height="150" width="219" x="472" y="100" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-12" parent="NAadGAteFJgWO_cu3Jsl-11" style="text;whiteSpace=wrap;html=1;" value="&lt;div style=&quot;background-color:#ffffff;color:#080808&quot;&gt;&lt;pre style=&quot;font-family: &amp;quot;JetBrains Mono&amp;quot;, monospace;&quot;&gt;&lt;font style=&quot;font-size: 10px;&quot;&gt;start: &lt;span style=&quot;color: rgb(0, 0, 0);&quot;&gt;pd.Timestamp&lt;br&gt;&lt;/span&gt;end: &lt;span style=&quot;color: rgb(0, 0, 0);&quot;&gt;pd.Timestamp&lt;br&gt;&lt;/span&gt;target_scale: &lt;span style=&quot;color: rgb(0, 0, 0);&quot;&gt;float&lt;/span&gt;&lt;/font&gt;&lt;/pre&gt;&lt;/div&gt;" vertex="1">
          <mxGeometry height="68.18181818181816" width="106.18181818181819" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-13" parent="NAadGAteFJgWO_cu3Jsl-11" style="text;whiteSpace=wrap;html=1;" value="&lt;div style=&quot;background-color:#ffffff;color:#080808&quot;&gt;&lt;pre style=&quot;font-family: &amp;quot;JetBrains Mono&amp;quot;, monospace;&quot;&gt;&lt;font style=&quot;font-size: 10px;&quot;&gt;X_in_sample: &lt;span style=&quot;color: rgb(0, 0, 0);&quot;&gt;pd.DataFrame&lt;br&gt;&lt;/span&gt;X_out_sample_last_index:&lt;/font&gt;&lt;/pre&gt;&lt;/div&gt;" vertex="1">
          <mxGeometry height="54.54545454545453" width="219.00000000000003" y="50.00454545454555" as="geometry" />
        </mxCell>
        <mxCell id="NAadGAteFJgWO_cu3Jsl-14" parent="NAadGAteFJgWO_cu3Jsl-11" style="text;whiteSpace=wrap;html=1;" value="&lt;div style=&quot;background-color:#ffffff;color:#080808&quot;&gt;&lt;pre style=&quot;font-family: &amp;quot;JetBrains Mono&amp;quot;, monospace;&quot;&gt;&lt;font style=&quot;font-size: 10px;&quot;&gt;DANGEROUS_r_next_in_sample: &lt;/font&gt;&lt;span style=&quot;color: rgb(0, 0, 0);&quot;&gt;&lt;span style=&quot;font-size: 9.8pt;&quot;&gt;&lt;br&gt;&lt;/span&gt;&lt;/span&gt;&lt;/pre&gt;&lt;/div&gt;" vertex="1">
          <mxGeometry height="47.72727272727272" width="212.36363636363637" y="102.27272727272724" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

